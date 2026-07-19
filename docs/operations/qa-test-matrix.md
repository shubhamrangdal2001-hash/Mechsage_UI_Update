# MechSage automated test matrix

Status at the start of the attached QA specification audit.

| Area | Existing executable coverage | Identified gaps |
|---|---|---|
| Static/CI | Python compile, ESLint, TypeScript, build, YAML, Compose | Ruff/coverage unavailable locally; no Playwright CI |
| Backend API | Health, auth, CORS, correlation ID, basic inference/work orders | Negative schemas, unknown IDs, transition conflicts, unavailable dependencies |
| Database | Isolated SQLite CRUD and approval | Rejection event, filters/pagination, rollback, live-alert dedup/reactivation/concurrency |
| ML | Eight artifacts load; four real synthetic-to-model predictions | Warm-up, repeatability, changing cycles, trigger/severity invariants, version and RUL bounds |
| RAG | Production API grounded query/abstention; legacy RAG unit suite | Four asset regressions, citations/corpus metadata, corrupt/empty corpus, turbofan regression |
| REST/WebSocket | Snapshot and control envelopes | Advancing real cycle, reconnect/stale/slow clients, once-per-transition alerts |
| Schema drift | OpenAPI route list and field-name presence | Enum/nullability/strictness and generated type validation |
| Frontend unit | MetricCard | Stores, cards, empty/error/disconnected states, filters, RAG, forms, controls |
| Browser E2E | Fleet render and asset detail navigation | RAG, work orders, agent, disconnect/recovery, live cycle assertions |
| Security | API key tests, CodeQL/gitleaks workflows | Dependency vulnerability remediation and production secret/config enforcement |

Canonical commands:

```powershell
python -m pytest backend/tests -q --basetemp .pytest_tmp/backend
python -m pytest dev/rag/tests -q --basetemp .pytest_tmp/rag
cd frontend
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
npm.cmd run test:e2e
```
