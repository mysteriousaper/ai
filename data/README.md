# data/

Everything in this folder except this README is git-ignored.

- `raw/`: datasets and lexicons exactly as downloaded by `scripts/fetch_data.py` (Phase 1).
- `processed/`: derived files (cleaned splits, merged lexicons, embeddings).

Licences for every file are tracked in `docs/DATA_SOURCES.md`. Personal study data (Phase 5)
never lives here; it goes to local SQLite or your own Supabase project.
