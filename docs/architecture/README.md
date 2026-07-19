# Architecture

MechSage is a monorepo with four stable runtime boundaries:

- `frontend/`: Next.js operations console.
- `backend/`: FastAPI contracts, persistence, and orchestration.
- `inference/`: feature construction and versioned ML inference.
- `dev/rag/` and `dev/agentic/`: grounded retrieval and diagnostic workflows.

The existing paths remain stable because Python imports, Docker build contexts,
tests, and model tooling depend on them. New modules should respect these
boundaries rather than introducing additional root-level applications.

Production requests flow from REST/WebSocket clients through the backend service
layer. Synthetic telemetry is transformed by the feature builder and evaluated
by versioned RUL and anomaly artifacts. Alerts create human-reviewable work-order
drafts. RAG supplies Ironside-specific citations; it never authorizes maintenance.
