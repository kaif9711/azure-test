Deprecated / Removed Frontend Assets
===================================

This file records intentional removals to keep the codebase aligned with the current roadmap (FastAPI backend + single static UI).

Removed 2025-10-09:

1. Lessons feature
   - Files: `js/lessons.js`, `lessons/en.json`, `lessons/ar.json`
   - Reason: Feature was an experimental educational/demo module not present in the revised product scope. Navigation and section already removed from `index.html`; files now deleted to reduce noise.
   - Replacement: None. If lightweight content pages are needed later, re‑introduce via a CMS or markdown-driven help panel.

Impact / Safety:
   - No runtime references remained (confirmed via search for `lessons` excluding historical comments in `main.js`).
   - Browser console will no longer request `lessons/*.json` or `lessons.js`.

If you need to restore these assets, recover them from git history (commit prior to this deletion).

2. Frontend i18n system
   - Files: `js/i18n.js`, `i18n/en.json`, `i18n/ar.json` (and all `data-i18n*` attributes & language dropdown removed from `index.html`; docker compose volume `./frontend/i18n` removed)
   - Reason: Multi-language support not in current product scope; keeping it added maintenance overhead and increased bundle surface.
   - Replacement: Static English text baked into HTML / JS. To reintroduce later, prefer a lightweight build‑time extraction (e.g. a JSON dictionary consumed by a minimal translator) or integrate a SaaS localization platform.
   - Safety Verification: Searched for `I18n`, `data-i18n`, `i18n/` after removal; only historical references existed inside deleted file. Runtime conditionals now simplified to plain strings.


Next Suggested Cleanups (pending confirmation):
Completed 2025-10-09 (Phase 2 Cleanup):
   - Removed legacy Flask routes (`backend/routes/*.py`) and `backend/app.py`.
   - Removed onboarding/demo page `demo.html` (superseded by main UI in `frontend/index.html`).
   - Removed unused generic backend `Dockerfile` (retained `Dockerfile.api` and `Dockerfile.ml`).
   - Removed `streamlit_dashboard` service from `docker-compose.yml` (feature not on roadmap).

Remaining Candidates (evaluate before removal):
   - Demo FastAPI variants: `minimal_api.py`, `simple_api.py` (confirm no tooling depends on them).
   - `backend/utils/monitor.py` (Flask-specific sections) – refactor or delete if metrics migrated.
   - Any README references to removed assets (update accordingly).

Approve those before proceeding for irreversible cleanup.
