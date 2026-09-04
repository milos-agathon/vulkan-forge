from __future__ import annotations

import copy
import os

import pytest

from forge3d.anamnesis import _Store, explain, render_sequence


def _recipe(text: str) -> dict:
    return {
        "terrain": {"dem_sha256": "11" * 32, "z_scale": 1.5},
        "atmosphere": {"model": "clear"},
        "lighting": {"azimuth": 215.0, "elevation": 35.0},
        "camera": {"path": "deterministic-flythrough-v1"},
        "layers": [
            {
                "kind": "label",
                "id": "summit",
                "text": text,
                "visible_frames": list(range(250, 260)),
            }
        ],
        "output": {"width": 64, "height": 64, "samples": 4},
    }


@pytest.mark.slow
def test_600_frame_incremental_matches_cold_and_recomputes_exact_set(tmp_path):
    warm_store = tmp_path / "warm"
    cold_store = tmp_path / "cold-modified"
    original = _recipe("Old summit")
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"

    render_sequence(original, cache=warm_store)
    incremental = render_sequence(modified, cache=warm_store)
    cold_modified = render_sequence(modified, cache=cold_store)

    assert incremental.frame_hashes == cold_modified.frame_hashes
    assert len(incremental.frame_hashes) == 600
    assert incremental.prediction_matches
    assert set(incremental.predicted_recompute) == set(incremental.observed_recompute)
    assert len(incremental.observed_recompute) == 21
    labels = {label for _, label in incremental.observed_recompute}
    assert labels == {"label.compile", "label.composite", "frame.output"}
    assert incremental.cache_report.hit_rate > 0.99


def test_dry_run_predicts_without_writing(tmp_path):
    store = tmp_path / "dry"
    result = render_sequence(_recipe("Summit"), frames=range(3), cache=store, dry_run=True)
    assert result.predicted_recompute
    assert result.observed_recompute == []
    assert not any(path.name == "blob" for path in store.rglob("blob"))


def test_explain_reconstructs_recursive_pass_derivation(tmp_path, capsys):
    result = render_sequence(_recipe("Summit"), frames=[250], cache=tmp_path)
    key = result.pass_keys["250:frame.output"]

    tree = explain(key, tmp_path)
    capsys.readouterr()

    assert tree["key"] == key
    assert tree["reconstructed_key"] == key
    assert tree["reconstructs"] is True
    assert tree["pipeline_descriptor_hex"]
    assert tree["uniform_hex"]
    assert tree["capability_fingerprint_hex"]
    assert tree["engine_fingerprint_hex"]
    assert tree["inputs"]
    assert all(item["binding"] and item["derivation"] for item in tree["inputs"])


def test_structured_scheduler_executes_only_changed_passes(tmp_path):
    original = _recipe("Old summit")
    original["layers"][0]["visible_frames"] = [250]
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"
    calls: list[tuple[str, int]] = []

    def identity(state, frame, inputs):
        calls.append(("identity", frame))
        return inputs[0] if inputs else repr(state).encode("utf-8")

    executors = {
        "terrain.shade": identity,
        "accumulation": identity,
        "label.compile": identity,
        "label.composite": identity,
        "frame.output": identity,
    }
    fingerprints = {
        label: f"test-executor:{label}".encode("utf-8") for label in executors
    }
    options = {
        "frames": [250],
        "verify_reads": False,
        "pass_executors": executors,
        "pass_executor_fingerprints": fingerprints,
        "pass_executor_contexts": {
            label: f"test-context:{label}".encode("utf-8")
            for label in executors
        },
    }

    render_sequence(original, cache=tmp_path / "warm-speed", **options)
    calls.clear()
    incremental = render_sequence(modified, cache=tmp_path / "warm-speed", **options)
    incremental_calls = list(calls)
    calls.clear()
    cold = render_sequence(modified, cache=tmp_path / "cold-speed", **options)

    assert incremental.frame_hashes == cold.frame_hashes
    assert incremental.prediction_matches
    assert incremental_calls == [
        ("identity", -1),
        ("identity", 250),
        ("identity", 250),
    ]


def test_incremental_elides_unchanged_ancestors_and_batches_lru_touches(
    tmp_path, monkeypatch
):
    original = _recipe("Old summit")
    original["layers"][0]["visible_frames"] = [5]
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"
    frames = range(12)

    render_sequence(original, frames=frames, cache=tmp_path, verify_reads=False)
    metadata_mtimes = {
        path: path.stat().st_mtime_ns for path in tmp_path.glob("??/*/meta.json")
    }
    read_keys: list[str] = []
    original_get = _Store.get

    def recording_get(self, key, **kwargs):
        read_keys.append(key)
        return original_get(self, key, **kwargs)

    monkeypatch.setattr(_Store, "get", recording_get)
    incremental = render_sequence(
        modified,
        frames=frames,
        cache=tmp_path,
        verify_reads=False,
    )

    # Eleven unchanged terminal frames plus the unchanged accumulation input
    # to the one changed label composite. Shadow/shade ancestors are committed
    # transitively by those keys and require no filesystem read.
    assert len(read_keys) == 12
    assert incremental.prediction_matches
    assert (tmp_path / "access.log").is_file()
    assert all(
        path.stat().st_mtime_ns == modified_at
        for path, modified_at in metadata_mtimes.items()
    )


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_real_gpu_600_frame_acceptance(tmp_path):
    from anamnesis_gpu_acceptance import run_acceptance

    result = run_acceptance(tmp_path)
    assert result["matching_frames"] == 600
    assert result["symmetric_difference"] == []
    assert result["speedup"] >= 20.0
