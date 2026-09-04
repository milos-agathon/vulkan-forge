# 14 — CENSOR: The Self-Auditing Engine — a signed certificate that no fallback ran

**Track:** Hybrid (execution-honesty keystone)  ·  **Depends on:** none. Enables #17 (ANAMNESIS, needs the capability/shader hash set), #19 (TESSELLA, needs real timestamps + BC/bindless features), and hardens #04 (TERRA-DETERMINATA) and #08 (VERITAS).

**Validation policy (superseding the original routine-CI language below):**
`docs/censor-validation-policy.md` is authoritative. CENSOR is a small
architectural truth contract. Routine pull requests run focused invariant
checks; complete test/feature matrices, production signing, all-golden
rendering, physical-GPU proof, and scratch red-proof corruption are explicit
acceptance or release evidence only. A downstream dependency on CENSOR does not
transitively inherit those heavyweight gates.

You are working in the local `forge3d` repository.

## Objective
forge3d's central engineering claims — "512 MiB host-visible budget", "GPU timing", "no silent placeholder fallbacks" — are, on inspection, unenforced. One line requests **zero** GPU features, which structurally guarantees the ~450-line GPU-timing subsystem threaded through ten modules can never emit a timestamp. `check_budget` has exactly **one** caller in the entire crate, and its default policy returns `Ok(())` on overage. Roughly **190** raw `device.create_buffer(` sites bypass the tracker; one goes through it. CI passes four Cargo features to `cargo test` that are referenced **nowhere** in `src/`, making the build look like it exercises IBL, cascaded shadows, render bundles, and memory pools. Twelve of 243 pytest files run in CI. Visual goldens cannot fail.

This is an IMPLEMENTATION task. Build the machine that makes those claims true and, more importantly, makes them *checkable by a stranger*: a negotiated GPU capability set, a total allocation ledger with enforce-by-default budgeting, live timestamp queries, and a deterministic `RenderCertificate` emitted with every render that names the adapter, the exact WGSL module hashes, the negotiated features, the per-pass timings, the peak allocation, and — the load-bearing field — a `degradations: []` array that must be empty. Routine CI protects these architectural invariants with focused source and behavior tests. Signing and exhaustive execution evidence certify an explicit acceptance/release candidate; they are not prerequisites for unrelated pull requests. A 3D map studio like no other cannot merely *be* correct; it must be unable to misreport how it rendered.

## Context — what exists today (verified)
- `src/core/gpu.rs:199` — `required_features: wgpu::Features::empty()` — **no GPU feature is ever requested.** Consequence chain: `src/core/gpu_timing.rs:127` tests `features.contains(Features::TIMESTAMP_QUERY)`, so `is_supported()` is permanently `false`; `encoder.write_timestamp` at `:261` is unreachable. Ten modules (`hdr`, bloom, postfx, ssgi, `terrain/lod`, `vector/indirect/culling`, …) thread a `GpuTimingManager` that cannot work.
- `src/core/device_caps.rs:82` — `descriptor_indexing` probe (`TEXTURE_BINDING_ARRAY` + non-uniform indexing) — probed **for reporting only** and never requested by the default device. A terrain LUT texture-array bind-group branch exists in `src/terrain/pipeline/creation.rs`, but it is dead while the main context requests `Features::empty()`. `:87` probes `FLOAT32_FILTERABLE` the same way.
- `src/core/memory_tracker/registry.rs:5` — `MEMORY_BUDGET_LIMIT: u64 = 512 * 1024 * 1024`; `:44` — `budget_policy: AtomicU8::new(BUDGET_POLICY_WARN)` — **default is warn**.
- `src/core/memory_tracker/reporting.rs:50` — `check_budget()` — the only enforcement point in the crate. On overage with the default policy it `log::warn!`s and `return Ok(())` (`:58-59`). Its **only caller** is `src/core/compressed_textures/upload.rs:21`.
- `src/core/memory_tracker/registry.rs:77` / `:110` — `track_buffer_allocation` / `track_texture_allocation` — pure counters, no gate.
- `src/core/resource_tracker.rs:62` — `tracked_create_buffer()` — the opt-in wrapper. Roughly **190** raw `device.create_buffer(` sites in `src/` vs **1** call to this wrapper outside its own file.
- `src/core/gpu.rs:231` — `Err(err) => panic!("{err}")` — the one non-test `panic!` in `src/`: adapter/device init aborts the **process** instead of returning `RenderError`. Every GPU entrypoint inherits it.
- `src/core/gpu.rs:19` — `static CTX: OnceCell<GpuContext>` — device created once, never re-created. Zero hits for `on_uncaptured_error`, `DeviceLostReason`, `push_error_scope`: a lost device is terminal.
- `src/core/gpu.rs:38` — `deterministic_mode()`; `:50` `requested_backend_from_env()`; `:125` `Dx12Compiler::Fxc` — the only shader-codegen determinism knob. `src/core/session.rs:39` — the `backend` argument warns `"not yet implemented"` and is discarded.
- `src/core/framegraph_impl/mod.rs:58` — `FrameGraph`; `:149` `compile()` — instantiated in exactly **one** place: `src/py_functions/diagnostics.rs:34`. The viewer and offscreen renderers hand-roll their passes. `src/core/framegraph.rs:43` — `add_gi_passes()` is a documented no-op.
- `src/core/postfx/chain.rs:124` — `execute_chain()` — **zero callers**. `src/core/postfx/mod.rs:19` — `postfx_apply_noop()`, an empty body.
- `src/render/memory_budget.rs` — `GpuMemoryBudget::new` referenced only inside its own `#[cfg(test)]` block (lines 397-455). A second, unrelated "memory budget" (`crate::util::memory_budget`) is what `src/render/material_set/gpu.rs:120` actually uses. Neither is the 512 MiB tracker.
- `Cargo.toml` declares **25** features. Six are referenced **nowhere** in `src/`, `tests/`, `benches/`, `build.rs`: `terrain_spike`, `exr`, `enable-ibl`, `enable-csm`, `enable-render-bundles`, `enable-memory-pools`. Four of those six are passed to `cargo check`/`cargo test` at `.github/workflows/ci.yml:105` and `:108`.
- `.github/workflows/ci.yml:249-258` — `test-python` runs **12 of 243** test files (~4.9%). `ci.yml:152` — the interactive-viewer job is `continue-on-error: true`. `ci.yml:331/346` — the golden-image job probes with `continue-on-error`, then `echo`s a skip and reports **success**; `ci.yml:419` `ci-success` accepts `skipped` for it.
- Already available (no new deps): `sha2 = "0.10"`, `ed25519-dalek = "2"` (both already used by the bundle/provenance path), `serde`, `serde_json`. `python/forge3d/recipe_manifest.py:305` — `_canonical_json_value` / `manifest_to_json` — an existing byte-stable canonical-JSON encoder you must reuse, not re-invent.

## Operating rules
- Inspect the actual repository state; do not rely on memory. Start with `git status --short` and leave unrelated dirty files alone.
- For routine pull requests, run format/lint plus focused compile and tests selected from the changed surface. Build/install the native extension when a change crosses the Python/native boundary or needs runtime verification. Full release builds belong to acceptance/release validation.
- Rust lint gate: `cargo forge3d-clippy` (never plain `cargo clippy`). Format with `cargo fmt`.
- **No new dependencies.** Signing uses the in-tree `ed25519-dalek`; hashing uses the in-tree `sha2`; canonical JSON reuses `recipe_manifest.py`'s encoder (and its Rust counterpart if one exists — grep first).
- Register any native symbol in `src/py_module/*`, re-export in `__all__`, update `EXPECTED_FUNCTIONS`/`EXPECTED_CLASSES` in `tests/test_api_contracts.py`, and update the `.pyi`.
- Requesting a GPU feature must **never** hard-fail on an adapter that lacks it. Every optional feature degrades explicitly and the degradation is *recorded in the certificate*. A degradation that is not recorded is the bug this task exists to kill.
- Do not weaken any existing test to make the new gates pass. If a gate exposes a real defect, fix the defect.

## What to build

1. **HARD CORE — capability negotiation, and the end of `Features::empty()`.**
   Add `src/core/capabilities.rs` with a `CapabilitySet` describing what forge3d *wants* (`TIMESTAMP_QUERY`, `PIPELINE_STATISTICS_QUERY`, `TEXTURE_BINDING_ARRAY` + `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`, `TEXTURE_COMPRESSION_BC`, `FLOAT32_FILTERABLE`, `INDIRECT_FIRST_INSTANCE`), what it *requires*, and what it *got*. Rewrite `src/core/gpu.rs:197-205` to intersect wants with `adapter.features()`, request the intersection, and store the result in `GpuContext`.
   - Every capability that was wanted and not granted appends a structured entry to the context's `degradations` list: `{kind: "capability_absent", name, consequence}`.
   - `src/core/gpu_timing.rs:127` must now return `true` on any adapter that granted `TIMESTAMP_QUERY`, and per-pass timings must actually populate. Fix `:394`'s `poll(Maintain::Wait)` so resolving queries does not stall the frame (double-buffer the query set).
   - `src/core/gpu.rs:231`: replace `panic!` with a propagated `RenderError`; surface it as a Python exception. Add `device.on_uncaptured_error` and a `push_error_scope`/`pop_error_scope` around every pipeline creation, recording validation errors as degradations. Add a `DeviceLost` handler that marks the context poisoned so subsequent calls raise rather than UB.
   - `src/core/session.rs:39`: either honor `backend=` (by setting the env before first `ctx()` and asserting no context exists yet) or raise. The current "warn and discard" is exactly the class of lie this task removes.

2. **HARD CORE — the total allocation ledger.**
   - Extend `src/core/resource_tracker.rs` with `tracked_create_buffer` / `tracked_create_texture` wrappers that: assign a mandatory label, classify host-visible vs device-local from the usage flags, call `check_budget` **before** allocating, and register the allocation in a global `AllocationLedger` (label, bytes, kind, backtrace-free call-site via `#[track_caller]`).
   - Migrate all raw `device.create_buffer(` sites and every `create_texture(` site. Enforce with a source-level gate: `tests/test_allocation_gate.py` (or a Rust `#[test]` that reads the tree) asserting **zero** raw `device.create_buffer(` / `device.create_texture(` outside `src/core/resource_tracker.rs`. Allow an explicit, documented allowlist file with a reason per entry — and assert the allowlist is empty or shrinking.
   - Flip `src/core/memory_tracker/registry.rs:44` to `BUDGET_POLICY_ENFORCE` by default. Overage returns `Err` naming the offending label and the current top-5 consumers. Expose `forge3d.mem.budget_policy(...)` so a user may *opt in* to warn, never the reverse.
   - Delete or unify the dead `src/render/memory_budget.rs`. Two "memory budget" systems, neither of which is the tracker, is itself a degradation.
   - Cross-check invariant: the ledger's `sum(bytes)` must equal the wrapper's own allocation counter at frame end. Assert it in a debug build.

3. **HARD CORE — the `RenderCertificate`.**
   Add `src/core/certificate.rs`. Every render (offscreen, golden, MapScene, PT reference) emits a certificate:
   ```
   {
     "schema": "forge3d.render_certificate/1",
     "engine": {"version", "git_sha", "wgsl_module_hashes": {"<pipeline label>": "<sha256>"}},
     "adapter": {"vendor", "device", "backend", "driver_info"},
     "capabilities": {"requested": [...], "granted": [...], "limits": {...}},
     "passes": [{"label", "gpu_ms", "draw_calls", "pipeline_stats"?}, ...],
     "allocations": {"peak_host_visible_bytes", "peak_device_local_bytes", "by_label": {...}},
     "degradations": [],
     "signature": {"alg": "ed25519", "pubkey", "sig", "signed_fields": [...]}
   }
   ```
   - The signed payload **excludes** the nondeterministic block (`passes[].gpu_ms`) and includes everything else, canonicalized with the existing byte-stable encoder (`recipe_manifest.py:326 manifest_to_json`; mirror it in Rust or reuse). Same scene + same adapter ⇒ byte-identical signed payload.
   - WGSL module hashes are computed over the **preprocessed** source actually handed to naga, not the file on disk.
   - `degradations` is populated from a single global sink. Every existing "fallback"/"placeholder"/"synthetic"/"unsupported_option" path in Rust and Python must push a structured entry into it. Wire, at minimum: `src/geo/reproject.rs:129` (`ProjNotEnabled`), `src/core/compressed_textures/parsing.rs:48,61`, `src/loaders/ktx2/loader.rs:134,139`, `src/core/async_compute/mod.rs:31,43,54`, `python/forge3d/sdf.py:21` (CPU fallback), `python/forge3d/buildings.py:354,495` (empty geometry), `python/forge3d/lighting.py:63,778-885` (nine setters that `warnings.warn` and return success), `python/forge3d/path_tracing.py:451` (`_synthetic_basis`).
   - Ship a standalone verifier: `python -m forge3d.certificate verify cert.json --pubkey k.pub` — no GPU, no forge3d native module, offline. Mutating any signed byte must fail verification; mutating one WGSL source byte must change a module hash.

4. **Runtime honesty invariants and acceptance evidence.**
   Add `tests/test_no_silent_degradation.py` to the focused routine profile for source/contract invariants. Keep exhaustive execution in the explicit full acceptance profile:
   - a. Every golden/reference render's certificate has `degradations == []`. (If a scene legitimately degrades on a hosted runner — e.g. no `TIMESTAMP_QUERY` on a software adapter — it must be listed in a committed `tests/degradation_allowlist.toml` with `reason`, `owner`, and an `expires` date; an expired entry **fails**.)
   - b. Source gate: zero raw `device.create_buffer(`/`create_texture(` outside the tracker.
   - c. Feature gate: every feature declared in `Cargo.toml` is referenced by at least one `cfg(feature = "…")` in `src/`/`tests/`/`benches/`/`build.rs`. This deletes or resurrects `terrain_spike`, `exr`, `enable-ibl`, `enable-csm`, `enable-render-bundles`, `enable-memory-pools`. Routine CI verifies declaration/list consistency and the affected compile surface. The full platform/feature matrix is acceptance/release evidence — no lane may advertise CSM coverage that does not exist.
   - d. Wheel gate: every feature referenced by `src/` and required by a *documented* public API is present in `pyproject.toml`'s maturin feature list, or the API raises a `DegradedCapability` error. (Today `proj` and `geos-topology` are referenced by 13 and 18 sites and are absent from the wheel.)
   - e. Test-coverage gate: every file matching `tests/test_*.py` remains tracked and assigned to the full acceptance profile, an explicit lane, or `tests/UNRUN.toml` with `reason`, `owner`, `expires`. Expired ⇒ fail. The focused routine profile is a deliberate subset and must retain the mandatory CENSOR truth contracts; it need not execute the complete repository suite.
   - f. `.github/workflows/ci.yml`: golden routing must be able to fail. A focused routine test locks `passed`/`absent`/`failed` semantics without rendering the whole catalog. In an explicit acceptance run, a *probe-positive* mismatch fails acceptance; probe-negative emits an `ABSENT` marker and a crash remains fatal.

5. **Wake the dead structure, or bury it.** `PostFxChain::execute_chain` (`postfx/chain.rs:124`) has zero callers while `src/viewer/render/main_loop/postfx.rs` hand-rolls the same work and creates three bind groups per stage per frame (`:27,:103,:263`), plus `tonemap.rs:345` allocating a bind group inside `render()`. Either route the viewer through the chain (and cache bind groups) or delete the chain and `postfx_apply_noop`. Same call for `RenderBundleManager` (`src/core/render_bundles.rs`, zero external callers) and `src/core/framegraph.rs:43`. Whatever survives must appear in a certificate `passes[]` entry. Record the decision in `AGENTS.md`.

## Public API / shader changes
- **Rust:** new `src/core/capabilities.rs`, `src/core/certificate.rs`, `src/core/degradation.rs` (the global sink). `src/core/gpu.rs` rewritten around negotiation + error scopes + device-lost. `src/core/resource_tracker.rs` gains `tracked_create_texture`, `#[track_caller]`, and the ledger. `src/core/memory_tracker/registry.rs:44` default flipped.
- **PyO3 / Python:** `forge3d.diagnostics.render_certificate() -> dict`; `forge3d.certificate.verify(path, pubkey) -> bool` (pure Python, importable without the native module); `forge3d.mem.budget_policy(policy)`; `forge3d.diagnostics.capabilities() -> dict`. Every render entry point gains `certificate=True|False|path`.
- **WGSL:** none. But every pipeline creation site must pass a stable `label` — the certificate keys on it.
- **CI:** `.github/workflows/ci.yml` job gating rewritten per 4.f; feature list corrected per 4.c.
- **Feature flags:** net **−6** (dead ones removed or wired). Add none.

## Definition of done
**MEASURABLE CONTRACT AND ACCEPTANCE CRITERIA — all seven:**

The tags state what routine work must preserve and what is swept only for a
named acceptance/release candidate.
1. **[focused routine contract; full empirical sweep at acceptance]** `Features::empty()` is gone. On a CI runner that grants `TIMESTAMP_QUERY`, `render_certificate()["passes"]` contains **≥ 5 passes with non-zero `gpu_ms`**. On one that does not, `degradations` contains exactly one `capability_absent` entry naming it, and the allowlist accepts it.
2. **[routine invariant; exhaustive execution at acceptance]** **Zero** raw `device.create_buffer(` / `device.create_texture(` outside `src/core/resource_tracker.rs` (currently roughly 190 / many). Gate is a test, not a lint suggestion.
3. **[routine invariant; exhaustive execution at acceptance]** Budget policy defaults to **enforce**. A scene that requests 600 MiB host-visible raises `MemoryBudgetExceeded` naming the offending label and the top-5 consumers — and the process does not abort. Ledger sum == wrapper counter (debug assert).
4. **[focused routine tamper contract; signed-render sweep at acceptance]** **Certificate determinism + tamper-evidence:** the signed payload for a fixed golden scene is byte-identical across two runs on the same adapter (SHA-256 equal); flipping one byte of any WGSL source changes exactly one `wgsl_module_hashes` entry and makes the golden gate fail; flipping one byte of the certificate makes `forge3d.certificate.verify` return `False`.
5. **[routine degradation contract; golden sweep at acceptance]** `degradations == []` for **every** committed golden render, or the exception is in `tests/degradation_allowlist.toml` with a non-expired date. `python/forge3d/lighting.py`'s nine "warn and report success" setters, `sdf.py`'s always-on CPU fallback, and `buildings.py`'s empty-geometry return either work or appear as degradations — no third option.
6. **[routine invariant; exhaustive execution at acceptance]** **Zero dead Cargo features**; `ci.yml`'s feature list equals the referenced set; `pyproject.toml`'s maturin features cover every capability a documented public API promises, or that API raises `DegradedCapability`.
7. **[routine accounting/routing contract; attributable proof at acceptance]** **Zero unaccounted test files:** `set(glob("tests/test_*.py")) == set(full_acceptance_profile) | set(explicit_lanes) | set(UNRUN.toml)`, no quarantined file is simultaneously claimed by an explicit lane, and no `UNRUN.toml` entry is expired. Golden routing must have a non-destructive negative-control test. Reuse the latest attributable scratch red-proof while the mechanism is unchanged; generate a new corruption run only for explicit acceptance/release after changing golden comparison, probe routing, or aggregate enforcement.

Plus: adapter-init failure raises a Python exception instead of aborting the process (test with `WGPU_BACKENDS=nonexistent`); `cargo forge3d-clippy` clean; `cargo fmt` applied.

## Tests & validation

**Routine pull-request minimum:** `cargo fmt --check`, `cargo forge3d-clippy`,
one affected compile/build, and the focused capability, degradation, allocation,
budget, certificate-contract, tamper-rejection, and workflow-policy tests.

**Acceptance/release:** release native build, curated cross-platform Rust/feature
matrix, the complete Python lanes selected by `scope=full`, candidate-selected goldens,
physical-GPU probes, and a production-key certificate sweep.
- Add: `tests/test_render_certificate.py`, `tests/test_certificate_verifier.py` (offline, no native module), `tests/test_no_silent_degradation.py`, `tests/test_allocation_gate.py`, `tests/test_capability_negotiation.py`, `tests/test_budget_enforce.py`, `tests/test_device_init_failure.py`.
- Add: `tests/UNRUN.toml`, `tests/degradation_allowlist.toml`, `tests/golden/certificates/*.json`.
- Rust: unit tests for `CapabilitySet` intersection, canonical-payload byte-stability, and the ledger invariant.
- Routine commands:
  - `cargo fmt --check` · `cargo forge3d-clippy` · one affected compile/build
  - `python scripts/ci_pytest_lane.py --profile fast -v --tb=short`
  - focused changed-surface tests, including the pure offline verifier when certificate behavior changes
- Acceptance/release commands:
  - release native/wheel builds and `cargo test --workspace --features <curated acceptance list> -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile`
  - `python scripts/ci_pytest_lane.py --profile full -v --tb=short` on the
    platforms selected by a manual `scope=full` run (report how many files execute)
  - candidate-selected GPU/golden probes and the production-key certificate sweep

## Non-goals
- Not per-pixel data provenance — that is VERITAS (#08). CENSOR certifies **execution**: which code ran, on which device, with which features, spending which bytes. VERITAS certifies **inputs**. They compose; do not merge them.
- Not cross-vendor bit-exactness — that is TERRA-DETERMINATA (#04). The certificate *records* the adapter; it does not promise identical pixels across adapters.
- Not a profiler UI, flamegraph, or Tracy integration. A JSON certificate plus pytest asserts is the whole surface.
- Do not add `lcms2`, `tracy-client`, `criterion` to the required path, or any new dependency.
- Do not delete failing tests, loosen goldens, or add `continue-on-error` to make the new gates green.
- Do not touch the unrelated in-flight dirty files.

## Verification before final response

For routine implementation work, confirm the focused truth contracts affected by
the change: status/diff, format/lint, relevant build, raw-allocation and feature
invariants, explicit degradation behavior, budget failure, certificate schema,
and tamper rejection. For an acceptance/release audit, additionally report full
matrix/suite counts, physical timing and golden results, production-signature
verification, and attributable red-proof evidence when the routing mechanism
changed. Missing acceptance infrastructure is `NOT_PROVEN`; it does not turn an
otherwise unrelated implementation pull request into a failure.
