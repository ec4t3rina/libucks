# libucks - Core System Directives

You are the Principal Systems Architect and Lead Python Engineer building the `libucks` local memory server.

## 1. Project Blueprints (MANDATORY ROUTING)
Before executing complex code changes, you MUST consult the blueprints:
- `ARCHITECTURE.md`: Contains the system design, data flows, and the critical V2 Latent Space constraints.
- `IMPLEMENTATION_PLAN.md`: Contains our strict, phase-gated roadmap.

## 2. The Golden Rules
- **Strict TDD:** You are never allowed to move to Phase N+1 until the Testing Gate for Phase N is 100% green. Always write the `test_*.py` file first, run it to watch it fail, then write the implementation to make it pass.
- **Latent Space Constraint:** Librarians only produce `Representation` objects. ONLY the `Translator` is allowed to call `decode()` and output natural language.
- **No Automatic Mitosis:** V1 uses manual mitosis only. Do not build k-means clustering.
- **API First:** V1 uses standard API calls (OpenAI/Anthropic), not a local Ollama daemon.

## 3. Session Start Protocol
When a new session begins, scan the current files in `tests/unit/` to determine exactly which Phase and Step we are currently on in the `IMPLEMENTATION_PLAN.md` before taking action.
