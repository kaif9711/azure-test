"""Document Validator Agent

Uses LayoutLMv3 / Donut / TrOCR to extract structured fields and compute document-level risk.
This is a scaffold; heavy models are loaded lazily. Add caching and batching in production.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import re

try:  # lightweight optional dependency for PDFs
    import PyPDF2  # type: ignore
except ImportError:  # pragma: no cover
    PyPDF2 = None

logger = logging.getLogger(__name__)

try:
    from transformers import (
        LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
        DonutProcessor, VisionEncoderDecoderModel,
        TrOCRProcessor
    )
    import torch
except ImportError:  # pragma: no cover - optional heavy deps already in requirements
    LayoutLMv3Processor = LayoutLMv3ForTokenClassification = DonutProcessor = VisionEncoderDecoderModel = TrOCRProcessor = object  # type: ignore
    torch = None

@dataclass
class DocumentField:
    name: str
    value: Any
    confidence: float
    page: Optional[int] = None

@dataclass
class DocumentValidationResult:
    extracted_fields: List[DocumentField]
    doc_risk: float
    issues: List[str]
    model_used: str
    timing_seconds: float

class DocumentValidatorAgent:
    """High-level document ingestion + extraction + risk scoring"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self._layoutlm = None
        self._layoutlm_processor = None
        self._donut = None
        self._donut_processor = None
        self._troc_encoder = None
        self._troc_processor = None

    # ----------------- Lazy loaders -----------------
    def _load_layoutlm(self):
        if self._layoutlm is None:
            logger.info("Loading LayoutLMv3 model...")
            self._layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
            self._layoutlm = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
            self._layoutlm.to(self.device)
        return self._layoutlm, self._layoutlm_processor

    def _load_donut(self):
        if self._donut is None:
            logger.info("Loading Donut DocVQA model...")
            self._donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
            self._donut = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa").to(self.device)
        return self._donut, self._donut_processor

    def _load_trocr(self):
        if self._troc_encoder is None:
            logger.info("Loading TrOCR model (printed)...")
            self._troc_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self._troc_encoder = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(self.device)
        return self._troc_encoder, self._troc_processor

    # ----------------- Public API -----------------
    def validate(self, file_path: str | Path) -> DocumentValidationResult:
        start = time.time()
        path = Path(file_path)
        issues: List[str] = []
        extracted: List[DocumentField] = []

        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        # Heuristic: if PDF or multi-page, prefer LayoutLM; else try Donut; fallback to OCR (TrOCR)
        suffix = path.suffix.lower()
        model_used = ""
        try:
            if suffix == ".pdf":
                model_used = "layoutlmv3"
                extracted.extend(self._extract_with_layoutlm(path))
            else:
                # Attempt Donut first
                try:
                    model_used = "donut"
                    extracted.extend(self._extract_with_donut(path))
                except Exception as e:  # fallback
                    issues.append(f"Donut failed ({e}); falling back to TrOCR")
                    model_used = "trocr"
                    extracted.extend(self._extract_with_trocr(path))
        except Exception as e:  # Last resort fallback OCR
            issues.append(f"Primary extraction failed ({e}); attempting TrOCR fallback")
            model_used = "trocr"
            try:
                extracted.extend(self._extract_with_trocr(path))
            except Exception as inner:
                issues.append(f"TrOCR fallback failed: {inner}")

        # Basic tamper / consistency heuristics (placeholder)
        doc_risk = self._compute_doc_risk(extracted, issues)

        return DocumentValidationResult(
            extracted_fields=extracted,
            doc_risk=doc_risk,
            issues=issues,
            model_used=model_used,
            timing_seconds=round(time.time() - start, 3)
        )

    # ----------------- Extraction Strategies -----------------
    def _extract_with_layoutlm(self, path: Path) -> List[DocumentField]:  # pragma: no cover (heavy)
        # For now, we attempt a lightweight PDF text extraction + regex pass before (optionally) invoking heavy model
        text_content = self._read_pdf_text(path)
        fields = self._regex_extract(text_content)
        # Only load heavy model if we plan to do token classification downstream (deferred placeholder)
        if not fields:  # fallback stub if regex failed
            fields.append(DocumentField(name="policy_number", value="UNKNOWN", confidence=0.40))
        return fields

    def _extract_with_donut(self, path: Path) -> List[DocumentField]:  # pragma: no cover (heavy)
        model, processor = self._load_donut()
        # Placeholder inference producing JSON-like tokens
        return [DocumentField(name="claim_amount", value=2500.0, confidence=0.81)]

    def _extract_with_trocr(self, path: Path) -> List[DocumentField]:  # pragma: no cover (heavy)
        model, processor = self._load_trocr()
        # Placeholder OCR single text line heuristic
        return [DocumentField(name="claimant_name", value="John Doe", confidence=0.75)]

    # ----------------- Risk Scoring -----------------
    def _compute_doc_risk(self, fields: List[DocumentField], issues: List[str]) -> float:
        if not fields:
            return 0.9  # High risk if nothing extracted
        avg_conf = sum(f.confidence for f in fields) / len(fields)
        penalty = 0.05 * len(issues)
        risk = 1 - max(0.0, min(1.0, avg_conf - penalty))
        return round(risk, 4)

    # ----------------- Utility helpers -----------------
    def _read_pdf_text(self, path: Path) -> str:
        if PyPDF2 is None:
            logger.warning("PyPDF2 not installed; cannot parse PDF deeply")
            return ""
        text_segments: List[str] = []
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_idx, page in enumerate(reader.pages[:10]):  # cap pages for speed
                    try:
                        text_segments.append(page.extract_text() or "")
                    except Exception as e:  # pragma: no cover
                        logger.debug(f"Page {page_idx} extraction failed: {e}")
        except Exception as e:  # pragma: no cover
            logger.debug(f"PDF read failed: {e}")
        return "\n".join(text_segments)

    def _regex_extract(self, text: str) -> List[DocumentField]:
        fields: List[DocumentField] = []
        if not text:
            return fields
        # Simple patterns (can be expanded)
        patterns = [
            ("policy_number", r"policy\s*(?:number|no)[:\-\s]*([A-Z0-9\-]{5,})", 0.85),
            ("claim_amount", r"claim\s*amount[:\-\s]*\$?([0-9]{2,}(?:\.[0-9]{2})?)", 0.8),
            ("claimant_name", r"claimant\s*(?:name)?[:\-\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)", 0.78),
            ("incident_date", r"incident\s*date[:\-\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})", 0.75),
        ]
        for name, pat, base_conf in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                fields.append(DocumentField(name=name, value=val, confidence=base_conf))
        return fields
