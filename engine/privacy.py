"""Privacy defaults (brief §1.8): no usage data leaves the machine via our dependencies.

Telemetry switches are set with ``setdefault`` so an explicit user setting always wins.
"""

import os

TELEMETRY_OFF: dict[str, str] = {
    "HF_HUB_DISABLE_TELEMETRY": "1",  # huggingface_hub / transformers
    "MLFLOW_DISABLE_TELEMETRY": "true",  # MLflow >= 3 usage telemetry
    "DO_NOT_TRACK": "1",  # generic opt-out honoured by MLflow and others
    "GRADIO_ANALYTICS_ENABLED": "False",  # Gradio demo UI (Phase 5)
}


def disable_third_party_telemetry() -> None:
    """Turn off library telemetry unless the user already chose a value."""
    for var, value in TELEMETRY_OFF.items():
        os.environ.setdefault(var, value)
