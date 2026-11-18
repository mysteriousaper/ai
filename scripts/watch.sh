#!/usr/bin/env bash
# Watch a PUBLIC video with the Gemini engine (ADR-0010).
#   scripts/watch.sh "<youtube-url>" --question "<what to look for>"
# Only the public URL and the question go to Google. Never pass user/study data here.
set -euo pipefail
export WATCH_ENGINE=gemini
# In cloud sessions the network proxy swaps this placeholder for the real key
# (x-goog-api-key header). Locally, set GEMINI_API_KEY yourself.
export GEMINI_API_KEY="${GEMINI_API_KEY:-proxy-injected-placeholder}"
exec python3 "$(dirname "$0")/../.claude/skills/watch/scripts/watch.py" "$@"
