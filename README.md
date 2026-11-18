# Emotion Inference Engine (MVP)

Text-first engine that estimates emotion (VAD + discrete labels), situation, need and temporal pattern, returned as a universal Emotional State Object. Full brief: [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

## Setup
```bash
uv sync        # installs Python 3.11 env + deps
make test      # smoke tests
make env       # print compute environment
```

Docs: [Architecture](docs/ARCHITECTURE.md) · [Decisions](docs/DECISIONS.md) · [Data sources](docs/DATA_SOURCES.md) · [Research](docs/RESEARCH.md) · [Tools](docs/TOOLS.md)

**Status:** Phase 2 complete. Architecture, schemas (`docs/schema/`), engine interfaces, mappings, safety rule, travel_v0 adapter; 75 tests green.
