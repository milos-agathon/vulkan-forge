from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def record_tessella_result(name: str, values: dict[str, Any]) -> None:
    """Print and, on the hardware lane, persist one verified numeric result."""
    payload = {"gate": name, **values}
    print(f"TESSELLA_RESULT {json.dumps(payload, sort_keys=True)}")
    artifact_dir = os.environ.get("FORGE3D_TESSELLA_ARTIFACT_DIR")
    if artifact_dir:
        destination = Path(artifact_dir) / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
