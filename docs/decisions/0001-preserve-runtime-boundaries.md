# ADR 0001: Preserve runtime directory boundaries

- Status: Accepted
- Date: 2026-07-19

## Context

Backend imports, model tooling, Docker contexts, and evaluation scripts currently
depend on the top-level `backend`, `frontend`, `inference`, and `dev` paths.
Moving them into a cosmetic `apps/` or `packages/` hierarchy would create a large,
high-risk change without improving runtime isolation.

## Decision

Keep the existing runtime directories and standardize the repository around them
with GitHub governance, focused workflows, documented ownership, reproducible
commands, and clear module boundaries.

## Consequences

The repository gains a conventional contributor experience while avoiding import
breakage. A future directory migration requires its own ADR and compatibility plan.
