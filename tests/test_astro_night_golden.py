"""SIDERA night-sky golden — DoD 6.

The frame must be byte-identical across two in-process renders and across two
*separate processes* on a pinned backend.  The committed NVIDIA/Vulkan reference must
also equal its PNG and SHA-256 sidecar in ``tests/goldens/determinism/`` — the
same inventory and zero-byte tolerance the TERRA-DETERMINATA harness
(``tests/test_determinism_hash.py``) applies to its reference renders.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import forge3d as f3d
from forge3d import _forge3d as native
from forge3d.diagnostics import render_certificate

if not f3d.has_gpu():
    pytest.skip(
        "the SIDERA night golden is a GPU render; no adapter present",
        allow_module_level=True,
    )


def _adapter_is_hardware() -> bool:
    """True only for a non-software adapter.

    ``has_gpu()`` is satisfied by WARP/lavapipe, which rasterise the blended
    sub-pixel star quads differently from real hardware. Following the
    ANAMNESIS rule in ``AGENTS.md``, a byte-exact comparison against a golden
    produced on one physical device is only claimed where the adapter proves it
    is not a software rasteriser. Hosted software/virtual adapters retain the
    in-process determinism coverage, while cross-process and committed-byte
    claims are ABSENT because deterministic mode intentionally refuses them.
    """
    try:
        probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    except Exception:  # pragma: no cover - probe failure means "cannot prove"
        return False
    if not isinstance(probe, dict):
        return False
    if bool(probe.get("software_fallback", False)):
        return False
    device_type = str(probe.get("device_type", "")).lower()
    name = str(probe.get("name", "")).lower()
    software_markers = (
        "cpu",
        "software",
        "warp",
        "lavapipe",
        "llvmpipe",
        "swiftshader",
        "paravirtual",
    )
    if any(marker in name for marker in software_markers):
        return False
    return device_type in {"discretegpu", "integratedgpu", "discrete_gpu", "integrated_gpu"}


def _adapter_is_nvidia_vulkan(probe: object) -> bool:
    if not isinstance(probe, dict) or probe.get("status") != "ok":
        return False
    name = str(probe.get("name", "")).lower()
    return (
        str(probe.get("backend", "")).lower() == "vulkan"
        and str(probe.get("device_type", "")).lower() == "discretegpu"
        and probe.get("software_fallback") is False
        and int(probe.get("vendor", 0)) == 0x10DE
        and "nvidia" in name
        and not any(
            marker in name
            for marker in ("software", "virtual", "paravirtual", "warp", "llvmpipe")
        )
    )


def _assert_expected_adapter(actual: dict) -> None:
    expected_path = os.environ.get("FORGE3D_EXPECTED_ADAPTER_PROBE")
    if not expected_path:
        return
    envelope = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    expected = envelope.get("probe")
    assert isinstance(expected, dict)
    for field in (
        "backend",
        "device_type",
        "name",
        "vendor",
        "device",
        "software_fallback",
    ):
        assert str(actual.get(field, "")).lower() == str(expected.get(field, "")).lower()


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "sidera_night.png"
#: Membership in the determinism harness' golden inventory.
GOLDEN_SHA = ROOT / "tests" / "goldens" / "determinism" / "sidera_night.sha256"
UPDATE_ENV = "FORGE3D_UPDATE_SIDERA_GOLDEN"


def _update_goldens_enabled() -> bool:
    """Read the refresh flag at *call* time.

    Binding this at import defeats per-test ``monkeypatch.delenv`` and has
    burned this repo before — see the CENSOR F-02 note in ``AGENTS.md``.
    """
    return os.environ.get(UPDATE_ENV) == "1"


def _determinism_backend() -> str:
    explicit = os.environ.get("FORGE3D_DETERMINISM_TEST_BACKEND")
    if explicit:
        return explicit
    if sys.platform == "win32":
        return "dx12"
    if sys.platform == "darwin":
        return "metal"
    return "vulkan"


REFERENCE_BACKEND = "vulkan"
_REFERENCE_SESSION = (
    f3d.Session(window=False)
    if _determinism_backend().strip().lower() == REFERENCE_BACKEND
    else None
)
REFERENCE_ADAPTER_INFO = f3d.device_probe("vulkan")
_assert_expected_adapter(REFERENCE_ADAPTER_INFO)
HARDWARE_ADAPTER = _adapter_is_hardware()
REFERENCE_ADAPTER = _adapter_is_nvidia_vulkan(REFERENCE_ADAPTER_INFO)
requires_hardware = pytest.mark.skipif(
    not HARDWARE_ADAPTER,
    reason="cross-process deterministic render requires a proven physical adapter",
)
requires_reference_hardware = pytest.mark.skipif(
    not REFERENCE_ADAPTER
    or _determinism_backend().strip().lower() != REFERENCE_BACKEND,
    reason="committed SIDERA bytes require a proven physical NVIDIA/Vulkan reference adapter",
)


def _render_in_subprocess(destination: Path) -> str:
    """Render the night golden in a fresh, backend-pinned process; return its SHA."""
    env = dict(os.environ)
    env.update(FORGE3D_DETERMINISTIC="1", WGPU_BACKENDS=_determinism_backend())
    env.pop(UPDATE_ENV, None)
    adapter_path = destination.with_suffix(".adapter.json")
    script = (
        "import json;"
        "from pathlib import Path;"
        "import forge3d as f3d;"
        "from forge3d import _forge3d as native;"
        "session=f3d.Session(window=False);"
        "probe=f3d.device_probe('vulkan');"
        f"Path({str(adapter_path)!r}).write_text(json.dumps(probe, sort_keys=True));"
        f"native._astro_night_golden_frame().save({str(destination)!r})"
    )
    subprocess.run(
        [sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True
    )
    subprocess_probe = json.loads(adapter_path.read_text(encoding="utf-8"))
    assert _adapter_is_nvidia_vulkan(subprocess_probe), subprocess_probe
    _assert_expected_adapter(subprocess_probe)
    artifact_dir = os.environ.get("FORGE3D_SIDERA_ARTIFACT_DIR")
    if artifact_dir:
        output = Path(artifact_dir) / "sidera-process-adapter.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(subprocess_probe, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _components(mask: np.ndarray) -> list[int]:
    pending = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while pending:
        stack = [pending.pop()]
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def _largest_component_bounds(mask: np.ndarray) -> tuple[int, int]:
    pending = set(map(tuple, np.argwhere(mask)))
    largest: list[tuple[int, int]] = []
    while pending:
        stack = [pending.pop()]
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        if len(component) > len(largest):
            largest = component
    assert largest
    ys, xs = zip(*largest)
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def test_night_sky_render_is_repeatable_and_certified(tmp_path):
    """Two in-process renders, zero byte tolerance, plus the certificate shape.

    Deliberately does NOT compare against the committed golden: the exact bytes
    depend on which backend wgpu selected for this process, and a test cannot
    pin that after the GPU context has been created. The committed-bytes claim
    therefore lives in the backend-pinned subprocess test below, which is how
    ``tests/test_determinism_hash.py`` makes the same kind of claim.
    """
    first = native._astro_night_golden_frame(certificate=True)
    second = native._astro_night_golden_frame(certificate=True)
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first.save(str(first_path))
    second.save(str(second_path))
    assert first_path.read_bytes() == second_path.read_bytes()

    rgba = np.asarray(first.to_numpy())
    assert rgba.shape == (512, 768, 4)
    # Moon + planets + stars: at least three lit, multi-pixel components.
    background = np.median(rgba[..., :3].reshape(-1, 3), axis=0).astype(np.int16)
    foreground = np.abs(rgba[..., :3].astype(np.int16) - background).max(axis=2) > 8
    assert len([size for size in _components(foreground) if size >= 2]) >= 3

    certificate = render_certificate(sign=False)
    labels = [entry["label"] for entry in certificate["passes"]]
    assert labels == ["astro.night.overlay", "astro.night.resolve"]
    assert certificate["models"] == {
        "astro.moonlight": (
            "Krisciunas and Schaefer 1991 phase and extinction; 0.172 mag per "
            "airmass; 0.25 lux pre-extinction reference; live terrain term "
            "normalized by the modeled zenith full Moon after extinction (not "
            "applied to this sky-only golden)"
        ),
        "astro.planet_display": (
            "artistic fixed display intensities; no physical planetary photometry claim"
        ),
        "astro.star_photometry": (
            "catalogue magnitude-to-irradiance ratio; resolution-invariant "
            "pixel-integrated tent splat in linear HDR; exposure 0.20; one "
            "Reinhard plus sRGB display resolve; no visibility floor"
        ),
        "astro.twilight": (
            "SIDERA civil-to-astronomical smoothstep; solar altitude -4 to -18 "
            "degrees; rendering visibility model"
        ),
    }
    assert set(certificate["engine"]["wgsl_module_hashes"]) == {
        "astro.night.resolve.shader",
        "astro.night.shader",
    }
    # `finish_render_capture` re-derives a `capability_absent` degradation for
    # every negotiated feature the device did not grant, and records
    # `timing_unavailable` when a timestamp query resolves invalid. Neither is
    # a SIDERA defect, so gate on the *kinds* the way the rest of the repo does
    # rather than demanding an empty list only an RTX-class device produces.
    granted = set(certificate["capabilities"]["granted"])
    for degradation in certificate["degradations"]:
        assert degradation["kind"] in {"capability_absent", "timing_unavailable"}, degradation
        if degradation["kind"] == "capability_absent":
            assert degradation["name"] not in granted, degradation
        else:
            assert degradation["name"] in {
                "astro.night.overlay",
                "astro.night.resolve",
            }, degradation


def test_night_frame_custom_aspects_preserve_the_moon_disc():
    """The dimension-specific projection does not stretch celestial discs."""
    for width, height in ((512, 256), (256, 512)):
        rgba = np.asarray(
            native._astro_night_golden_frame(width=width, height=height).to_numpy()
        )
        assert rgba.shape == (height, width, 4)
        background = np.median(rgba[..., :3].reshape(-1, 3), axis=0).astype(np.int16)
        foreground = np.abs(rgba[..., :3].astype(np.int16) - background).max(axis=2) > 20
        disc_width, disc_height = _largest_component_bounds(foreground)
        assert abs(disc_width - disc_height) <= 2, (width, height, disc_width, disc_height)


@requires_hardware
def test_night_golden_is_cross_process_repeatable_on_pinned_backend(tmp_path):
    """DoD 6: two processes on the selected backend produce identical bytes."""
    first = _render_in_subprocess(tmp_path / "process_a.png")
    second = _render_in_subprocess(tmp_path / "process_b.png")
    assert first == second, (
        "cross-process nondeterminism in the SIDERA night render on "
        f"{_determinism_backend()}\n  first:  {first}\n  second: {second}"
    )


@requires_reference_hardware
def test_night_golden_matches_committed_vulkan_bytes(tmp_path):
    """DoD 6: the pinned physical NVIDIA/Vulkan render equals the committed golden.

    This is the SIDERA member of the ``tests/test_determinism_hash.py`` family:
    same zero-byte tolerance and same committed SHA-256 sidecar. Other physical
    backends prove their own repeatability above; they are not claimed to
    reproduce the protected NVIDIA/Vulkan reference.
    """
    first_png = tmp_path / "process_a.png"
    first = _render_in_subprocess(first_png)

    if _update_goldens_enabled():
        # Refreshing happens here, from the pinned NVIDIA/Vulkan render, so the
        # committed bytes can never come from whatever backend the ambient
        # test process happened to pick.
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_bytes(first_png.read_bytes())
        GOLDEN_SHA.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_SHA.write_text(f"{first}\n", encoding="ascii")

    committed = hashlib.sha256(GOLDEN.read_bytes()).hexdigest()
    assert first == committed, (
        "night golden hash mismatch against the committed PNG\n"
        f"  golden: {committed}\n  actual: {first}\n"
        f"Zero-byte tolerance: regenerate deliberately with {UPDATE_ENV}=1 and "
        f"say why in the commit message."
    )
    assert GOLDEN_SHA.exists(), f"missing determinism sidecar {GOLDEN_SHA}"
    assert GOLDEN_SHA.read_text(encoding="ascii").split()[0].strip() == committed
    assert np.array_equal(f3d.png_to_numpy(first_png), f3d.png_to_numpy(GOLDEN))


@requires_reference_hardware
def test_golden_refresh_does_not_rewrite_the_committed_file_when_disabled(
    tmp_path, monkeypatch
):
    """Negative control for the refresh flag.

    Simulates a refresh run and then removes the flag; the committed golden and
    its sidecar must be untouched. If ``_update_goldens_enabled`` were bound at
    import, this ``delenv`` would be dead code and a corrupted frame could be
    copied over the golden during a refresh sweep.
    """
    before_png = GOLDEN.read_bytes()
    before_sha = GOLDEN_SHA.read_bytes()
    monkeypatch.setenv(UPDATE_ENV, "1")
    assert _update_goldens_enabled() is True
    monkeypatch.delenv(UPDATE_ENV)
    assert _update_goldens_enabled() is False
    test_night_golden_matches_committed_vulkan_bytes(tmp_path)
    assert GOLDEN.read_bytes() == before_png
    assert GOLDEN_SHA.read_bytes() == before_sha
