"""Fresh-process physical Metal probes for AETHER hard acceptance gates."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

from _aether_quadrature import physical_metal_probe, write_constant_hdr
from test_atmosphere_reference import (
    _make_metal_runtime,
    _measure_high_exposure_sky_hdr,
    _measure_terrain_exposure_scaling,
    _measure_terrain_saturation,
    _render_lut_sky_samples,
    _render_prometheus_reference_samples,
)


RESULT_PREFIX = "AETHER_PHYSICAL_RESULT="


def _require_physical_metal() -> dict:
    physical, probe = physical_metal_probe()
    if not physical:
        raise RuntimeError(f"physical Metal adapter required, got {probe}")
    return probe


def _sky_lut_probe(elevation: float) -> dict:
    probe = _require_physical_metal()
    with tempfile.TemporaryDirectory(prefix="aether-sky-") as directory:
        hdr_path = Path(directory) / "constant.hdr"
        write_constant_hdr(hdr_path)
        runtime = _make_metal_runtime(hdr_path)
        samples = _render_lut_sky_samples(runtime, elevation).tolist()
    return {
        "mode": "sky-lut",
        "probe": probe,
        "elevation": elevation,
        "samples": samples,
    }


def _sky_prometheus_probe(elevation: float) -> dict:
    probe = _require_physical_metal()
    samples, reference = _render_prometheus_reference_samples(elevation)
    return {
        "mode": "sky-prometheus",
        "probe": probe,
        "elevation": elevation,
        "samples": samples.tolist(),
        "reference": reference,
    }


def _saturation_probe() -> dict:
    probe = _require_physical_metal()
    with tempfile.TemporaryDirectory(prefix="aether-saturation-") as directory:
        hdr_path = Path(directory) / "constant.hdr"
        write_constant_hdr(hdr_path)
        runtime = _make_metal_runtime(hdr_path)
        measurement = _measure_terrain_saturation(runtime)
    return {"mode": "saturation", "probe": probe, "measurement": measurement}


def _exposure_probe() -> dict:
    probe = _require_physical_metal()
    with tempfile.TemporaryDirectory(prefix="aether-exposure-") as directory:
        hdr_path = Path(directory) / "constant.hdr"
        write_constant_hdr(hdr_path)
        runtime = _make_metal_runtime(hdr_path)
        measurement = _measure_terrain_exposure_scaling(runtime)
    return {"mode": "exposure", "probe": probe, "measurement": measurement}


def _high_exposure_probe() -> dict:
    probe = _require_physical_metal()
    with tempfile.TemporaryDirectory(prefix="aether-high-exposure-") as directory:
        hdr_path = Path(directory) / "constant.hdr"
        write_constant_hdr(hdr_path)
        runtime = _make_metal_runtime(hdr_path)
        measurement = _measure_high_exposure_sky_hdr(runtime)
    return {"mode": "high-exposure", "probe": probe, "measurement": measurement}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) >= 2 else ""
    if mode == "saturation" and len(sys.argv) == 2:
        result = _saturation_probe()
    elif mode == "exposure" and len(sys.argv) == 2:
        result = _exposure_probe()
    elif mode == "high-exposure" and len(sys.argv) == 2:
        result = _high_exposure_probe()
    elif mode in {"sky-lut", "sky-prometheus"} and len(sys.argv) == 3:
        elevation = float(sys.argv[2])
        result = (
            _sky_lut_probe(elevation)
            if mode == "sky-lut"
            else _sky_prometheus_probe(elevation)
        )
    else:
        raise SystemExit(
            "usage: _aether_physical_probe.py exposure|high-exposure|saturation|sky-lut DEG|sky-prometheus DEG"
        )
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
