"""
Supervisor Module
AI-powered supervisor for reviewing flagged claims and making final decisions
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import os
from enum import Enum

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    INVESTIGATE = "investigate"
    REQUEST_INFO = "request_additional_info"

class SupervisorAgent:
    """AI supervisor for claim review and decision making"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.approval_threshold = self.config.get('approval_threshold', 0.3)
        self.rejection_threshold = self.config.get('rejection_threshold', 0.8)
        
        # Decision rules and weights
        self.decision_rules = {
            'risk_score_weight': 0.4,
            'document_quality_weight': 0.3,
            'claim_history_weight': 0.2,
            'amount_threshold_weight': 0.1
        }
        
        # Thresholds for different actions
        self.thresholds = {
            'auto_approve': 0.2,
            'manual_review': 0.4,
            'investigation': 0.7,
            'auto_reject': 0.9
        }
    
    def review_claim(self, claim_data: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main supervisor review function
        
        Args:
            claim_data: Original claim information
            risk_assessment: Risk assessment results from RiskChecker
            
        Returns:
            Dictionary with supervisor decision and reasoning
        """
        try:
            # Analyze all available information
            analysis = self._comprehensive_analysis(claim_data, risk_assessment)
            
            # Make decision based on analysis
            decision = self._make_decision(analysis)
            
            # Generate detailed explanation
            explanation = self._generate_explanation(analysis, decision)
            
            # Calculate confidence in decision
            confidence = self._calculate_decision_confidence(analysis, decision)
            
            # Generate next steps
            next_steps = self._generate_next_steps(decision, analysis)
            
            result = {
                'decision': decision.value,
                'confidence': confidence,
                'explanation': explanation,
                'analysis_summary': analysis['summary'],
                'risk_factors': analysis['risk_factors'],
                'positive_factors': analysis['positive_factors'],
                'next_steps': next_steps,
                'review_timestamp': datetime.now().isoformat(),
                'requires_human_oversight': decision in [DecisionType.INVESTIGATE, DecisionType.REJECT] and confidence < 0.8
            }
            
            logger.info(f"Supervisor review completed: {decision.value} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in supervisor review: {str(e)}")
            return self._create_error_response(str(e))
    
    def _comprehensive_analysis(self, claim_data: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of the claim"""
        analysis = {
            'risk_factors': [],
            'positive_factors': [],
            'concerns': [],
            'strengths': [],
            'summary': {}
        }
        
        # Analyze risk score
        risk_score = risk_assessment.get('risk_score', 0.5)
        analysis['summary']['risk_score'] = risk_score
        
        if risk_score > 0.7:
            analysis['risk_factors'].append('High fraud risk score')
        elif risk_score < 0.3:
            analysis['positive_factors'].append('Low fraud risk score')
        
        # Analyze claim amount
        claim_amount = float(claim_data.get('claim_amount', 0))
        analysis['summary']['claim_amount'] = claim_amount
        
        if claim_amount > 100000:
            analysis['risk_factors'].append('Very high claim amount')
        elif claim_amount > 50000:
            analysis['concerns'].append('High claim amount requires verification')
        elif claim_amount < 1000:
            analysis['positive_factors'].append('Low claim amount')
        
        # Analyze claimant history
        previous_claims = int(claim_data.get('previous_claims', 0))
        analysis['summary']['previous_claims'] = previous_claims
        
        if previous_claims > 5:
            analysis['risk_factors'].append('Multiple previous claims')
        elif previous_claims == 0:
            analysis['positive_factors'].append('First-time claimant')
        
        # Analyze document quality
        doc_score = float(claim_data.get('document_confidence_score', 1.0))
        analysis['summary']['document_quality'] = doc_score
        
        if doc_score < 0.5:
            analysis['risk_factors'].append('Poor document quality')
        elif doc_score > 0.9:
            analysis['positive_factors'].append('High-quality documentation')
        
        # Analyze policy details
        policy_duration = int(claim_data.get('policy_duration_months', 12))
        analysis['summary']['policy_duration'] = policy_duration
        
        if policy_duration < 3:
            analysis['risk_factors'].append('Very new policy')
        elif policy_duration > 60:
            analysis['positive_factors'].append('Long-term policyholder')
        
        # Analyze temporal patterns
        self._analyze_temporal_patterns(claim_data, analysis)
        
        # Analyze claim-to-premium ratio
        policy_premium = float(claim_data.get('policy_premium', 1000))
        if policy_premium > 0:
            ratio = claim_amount / policy_premium
            analysis['summary']['claim_to_premium_ratio'] = ratio
            
            if ratio > 20:
                analysis['risk_factors'].append('Very high claim-to-premium ratio')
            elif ratio < 2:
                analysis['positive_factors'].append('Reasonable claim-to-premium ratio')
        
        return analysis
    
    def _analyze_temporal_patterns(self, claim_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Analyze temporal patterns in the claim"""
        try:
            claim_date = datetime.fromisoformat(claim_data.get('claim_date', datetime.now().isoformat()))
            
            # Weekend claims
            if claim_date.weekday() >= 5:
                analysis['concerns'].append('Weekend claim submission')
            
            # Late night claims
            if hasattr(claim_date, 'hour') and (claim_date.hour < 6 or claim_date.hour > 22):
                analysis['concerns'].append('Off-hours claim submission')
            
            # Recent claims pattern
            claims_last_30_days = int(claim_data.get('claims_last_30_days', 0))
            if claims_last_30_days > 1:
                analysis['risk_factors'].append('Multiple claims in last 30 days')
            
        except Exception as e:
            logger.warning(f"Error analyzing temporal patterns: {str(e)}")
    
    def _make_decision(self, analysis: Dict[str, Any]) -> DecisionType:
        """Make final decision based on analysis"""
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(analysis)
        
        # Decision logic
        if overall_risk >= self.thresholds['auto_reject']:
            return DecisionType.REJECT
        elif overall_risk >= self.thresholds['investigation']:
            return DecisionType.INVESTIGATE
        elif overall_risk >= self.thresholds['manual_review']:
            return DecisionType.REQUEST_INFO
        elif overall_risk <= self.thresholds['auto_approve']:
            return DecisionType.APPROVE
        else:
            return DecisionType.REQUEST_INFO  # Default to requesting more info
    
    def _calculate_overall_risk(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall risk score based on analysis"""
        base_risk = analysis['summary'].get('risk_score', 0.5)
        
        # Adjust based on risk factors
        risk_factor_penalty = len(analysis['risk_factors']) * 0.1
        positive_factor_bonus = len(analysis['positive_factors']) * 0.05
        concern_penalty = len(analysis['concerns']) * 0.05
        
        # Specific adjustments
        claim_amount = analysis['summary'].get('claim_amount', 0)
        if claim_amount > 100000:
            risk_factor_penalty += 0.2
        
        doc_quality = analysis['summary'].get('document_quality', 1.0)
        if doc_quality < 0.5:
            risk_factor_penalty += 0.15
        
        # Calculate final risk
        overall_risk = base_risk + risk_factor_penalty - positive_factor_bonus + concern_penalty
        return max(0.0, min(1.0, overall_risk))
    
    def _generate_explanation(self, analysis: Dict[str, Any], decision: DecisionType) -> str:
        """Generate human-readable explanation for the decision"""
        explanations = {
            DecisionType.APPROVE: "Claim approved based on low risk assessment and positive indicators.",
            DecisionType.REJECT: "Claim rejected due to high fraud risk indicators.",
            DecisionType.INVESTIGATE: "Claim requires investigation due to multiple risk factors.",
            DecisionType.REQUEST_INFO: "Additional information required before final decision."
        }
        
        base_explanation = explanations[decision]
        
        # Add specific details
        details = []
        
        if analysis['risk_factors']:
            details.append(f"Risk factors identified: {', '.join(analysis['risk_factors'])}")
        
        if analysis['positive_factors']:
            details.append(f"Positive indicators: {', '.join(analysis['positive_factors'])}")
        
        if analysis['concerns']:
            details.append(f"Concerns noted: {', '.join(analysis['concerns'])}")
        
        # Add risk score context
        risk_score = analysis['summary'].get('risk_score', 0.5)
        details.append(f"Overall risk score: {risk_score:.3f}")
        
        if details:
            return f"{base_explanation} {' '.join(details)}"
        else:
            return base_explanation
    
    def _calculate_decision_confidence(self, analysis: Dict[str, Any], decision: DecisionType) -> float:
        """Calculate confidence in the decision"""
        base_confidence = 0.7
        
        # Increase confidence for clear-cut cases
        risk_score = analysis['summary'].get('risk_score', 0.5)
        
        if decision == DecisionType.APPROVE and risk_score < 0.2:
            base_confidence = 0.95
        elif decision == DecisionType.REJECT and risk_score > 0.8:
            base_confidence = 0.95
        elif decision == DecisionType.INVESTIGATE and 0.6 <= risk_score <= 0.8:
            base_confidence = 0.85
        
        # Adjust based on data quality
        doc_quality = analysis['summary'].get('document_quality', 1.0)
        if doc_quality > 0.9:
            base_confidence += 0.05
        elif doc_quality < 0.5:
            base_confidence -= 0.1
        
        # Adjust based on consistency of indicators
        risk_factors_count = len(analysis['risk_factors'])
        positive_factors_count = len(analysis['positive_factors'])
        
        if risk_factors_count == 0 and positive_factors_count > 2:
            base_confidence += 0.1  # Consistently positive
        elif risk_factors_count > 3 and positive_factors_count == 0:
            base_confidence += 0.1  # Consistently negative
        elif risk_factors_count > 0 and positive_factors_count > 0:
            base_confidence -= 0.1  # Mixed signals
        
        return max(0.1, min(1.0, base_confidence))
    
    def _generate_next_steps(self, decision: DecisionType, analysis: Dict[str, Any]) -> List[str]:
        """Generate next steps based on decision"""
        next_steps = []
        
        if decision == DecisionType.APPROVE:
            next_steps = [
                "Process claim for payment",
                "Send approval notification to claimant",
                "Update claim status in system",
                "Archive case documentation"
            ]
        
        elif decision == DecisionType.REJECT:
            next_steps = [
                "Send rejection letter to claimant with detailed reasoning",
                "Document rejection rationale in system",
                "Provide appeal process information",
                "Flag account for future reference"
            ]
        
        elif decision == DecisionType.INVESTIGATE:
            next_steps = [
                "Assign to Special Investigation Unit (SIU)",
                "Schedule claimant interview",
                "Request additional documentation",
                "Verify incident details independently",
                "Review similar claims patterns"
            ]
        
        elif decision == DecisionType.REQUEST_INFO:
            next_steps = [
                "Contact claimant for additional information",
                "Specify required documents or clarifications",
                "Set follow-up timeline",
                "Place claim on hold pending response"
            ]
        
        # Add specific steps based on risk factors
        if 'Poor document quality' in analysis.get('risk_factors', []):
            next_steps.append("Request higher quality document scans")
        
        if 'High claim amount' in analysis.get('risk_factors', []):
            next_steps.append("Verify repair estimates or medical bills")
        
        if 'Multiple previous claims' in analysis.get('risk_factors', []):
            next_steps.append("Review complete claim history pattern")
        
        return next_steps
    
    def batch_review(self, claims_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Review multiple claims in batch"""
        results = []
        
        for claim in claims_batch:
            try:
                claim_data = claim.get('claim_data', {})
                risk_assessment = claim.get('risk_assessment', {})
                
                review_result = self.review_claim(claim_data, risk_assessment)
                review_result['claim_id'] = claim.get('claim_id')
                
                results.append(review_result)
                
            except Exception as e:
                logger.error(f"Error reviewing claim {claim.get('claim_id', 'unknown')}: {str(e)}")
                results.append(self._create_error_response(str(e)))
        
        return results
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update decision thresholds"""
        for key, value in new_thresholds.items():
            if key in self.thresholds and 0 <= value <= 1:
                self.thresholds[key] = value
                logger.info(f"Updated threshold {key} to {value}")
            else:
                logger.warning(f"Invalid threshold update: {key} = {value}")
    
    def get_decision_statistics(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from recent decisions"""
        if not decisions:
            return {}
        
        decision_counts = {}
        confidence_sum = 0
        
        for decision in decisions:
            decision_type = decision.get('decision', 'unknown')
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
            confidence_sum += decision.get('confidence', 0)
        
        total_decisions = len(decisions)
        avg_confidence = confidence_sum / total_decisions if total_decisions > 0 else 0
        
        return {
            'total_decisions': total_decisions,
            'decision_breakdown': decision_counts,
            'average_confidence': round(avg_confidence, 3),
            'approval_rate': decision_counts.get('approve', 0) / total_decisions * 100,
            'rejection_rate': decision_counts.get('reject', 0) / total_decisions * 100,
            'investigation_rate': decision_counts.get('investigate', 0) / total_decisions * 100
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response for failed reviews"""
        return {
            'decision': DecisionType.REQUEST_INFO.value,
            'confidence': 0.1,
            'explanation': f'Review error occurred: {error_message}. Manual review required.',
            'analysis_summary': {},
            'risk_factors': ['System error during review'],
            'positive_factors': [],
            'next_steps': ['Manual review required due to system error'],
            'review_timestamp': datetime.now().isoformat(),
            'requires_human_oversight': True,
            'error': error_message
        }