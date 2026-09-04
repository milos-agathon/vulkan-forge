from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from forge3d._png import load_png_rgba
from forge3d.helpers.offscreen import save_png_deterministic


RESULT_KEYS = (
    "normal_lighting_ssim",
    "family_residency_budget",
    "missing_family_fatal",
    "partial_normal_residency",
)


def _artifact_dir() -> Path | None:
    value = os.environ.get("FORGE3D_SUBSTRATIA_ARTIFACT_DIR")
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_substratia_result(name: str, values: Mapping[str, Any]) -> None:
    """Print and optionally persist one deterministic SUBSTRATIA gate result."""
    payload = {"gate": name, **dict(values)}
    print(f"SUBSTRATIA_RESULT {json.dumps(payload, sort_keys=True)}")
    artifact_dir = _artifact_dir()
    if artifact_dir is None:
        return
    ledger_path = artifact_dir / "results.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict):
            raise ValueError("SUBSTRATIA results ledger must be a JSON object")
    else:
        ledger = {
            "schema": "forge3d.substratia.results.v1",
            "candidate_sha": os.environ.get("FORGE3D_SUBSTRATIA_CANDIDATE_SHA", ""),
            "gates": {},
        }
    if ledger.get("schema") != "forge3d.substratia.results.v1":
        raise ValueError("SUBSTRATIA results ledger has an unexpected schema")
    gates = ledger.setdefault("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("SUBSTRATIA results ledger gates must be a JSON object")
    gates[name] = dict(values)
    ledger["gates"] = {key: gates[key] for key in sorted(gates)}
    temporary = ledger_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(ledger_path)


def record_substratia_image(filename: str, image: np.ndarray) -> Path | None:
    artifact_dir = _artifact_dir()
    if artifact_dir is None:
        return None
    path = artifact_dir / filename
    rgba = np.asarray(image)
    if rgba.dtype != np.uint8:
        rgba = np.clip(np.rint(rgba), 0, 255).astype(np.uint8)
    save_png_deterministic(path, np.ascontiguousarray(rgba))
    return path


def load_golden(path: Path) -> np.ndarray:
    if not path.is_file():
        raise AssertionError(
            f"required committed SUBSTRATIA golden is missing: {path}"
        )
    return np.asarray(load_png_rgba(path), dtype=np.uint8)


def image_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
