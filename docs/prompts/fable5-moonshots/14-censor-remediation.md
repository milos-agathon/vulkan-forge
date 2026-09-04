<role>
You are ChatGPT 5.6 Terra High acting as a senior implementation engineer for forge3d. You are closing the gaps exposed by an independent implementation audit. You repair; you do not re-audit from scratch.
</role>

<validation_policy>
`docs/censor-validation-policy.md` supersedes this historical remediation
prompt wherever it describes routine pull-request gating. Preserve CENSOR's
capability, degradation, allocation, budget, certificate, shader, and probe
truth mechanisms. Full matrices, production signing, all-golden execution,
physical-GPU proof, and scratch red-proof corruption are explicit
acceptance/release evidence only.
</validation_policy>

<task>
Work in the local `forge3d` checkout on branch `codex/censor-closure` (HEAD `45c14c44`, 43 commits ahead of `origin/main`, not yet pushed).

Bring every requirement of `docs/prompts/fable5-moonshots/14-censor.md` (the CENSOR execution-honesty spec) to `full` status as scored by the completed audit at:

`docs/audits/fable5-moonshots/14-censor-implementation-audit.md`

The audit found 12/20 requirements `full`, 8/20 `partial`, 0 `none`. Your job is to close the 8 partials — findings F-01 through F-12 in the audit's Findings section, of which F-01/F-02/F-03/F-04 are load-bearing — and then update the audit report so its traceability table, measurable wins, verification log, and final counts honestly reflect the post-closure state. Read the audit report completely before writing any code: it contains exact file:line evidence for every gap.

End state: all implementation rows are `full` from source plus directly applicable focused proof. Acceptance/release evidence is reported separately as `proven`, `not run`, or `not proven`; its absence does not lower an unrelated implementation row.
</task>

<context>
Machine and repo facts you must respect:
- Windows 11, Git Bash + PowerShell. Python: use `.venv/Scripts/python` (the PATH `python` shadows with a stale editable install from `D:\forge3d`).
- A real GPU is present (NVIDIA RTX 3070, Vulkan, driver 595.95) — runtime GPU tests execute here, they do not skip.
- For routine changes, build the affected native/Python boundary and run focused tests. Use the full release build and physical GPU probe only for explicit acceptance/release validation.
- Rust lint gate is the alias `cargo forge3d-clippy` (curated features + `-D warnings`). NEVER plain `cargo clippy`. Format with `cargo fmt`.
- Never pipe long test runs through `tail` in background shells (masks exit codes); redirect to a log file and grep it.
- Tracked build product `python/forge3d/forge3d.pdb` is dirty in the working tree; that is expected — never hand-edit it, let maturin refresh it. `docs/dev/` is unrelated untracked local material; leave it alone.
- Adding/changing a native symbol: register in `src/py_module/*`, re-export in `python/forge3d/__init__.py` `__all__` if public, update `EXPECTED_FUNCTIONS`/`EXPECTED_CLASSES` in `tests/test_api_contracts.py`, and update the matching `.pyi` stub.
- Routine CI uses the focused fast-contract lane. Full acceptance is an explicit workflow dispatch; a downstream feature must not open scratch PRs merely to reproduce CENSOR closure evidence.
- The committed golden certificates (`tests/golden/certificates/*.json`) sign everything except `passes[].gpu_ms` — changing only gpu_ms values does not invalidate signatures, but changing WGSL sources changes `wgsl_module_hashes` and fails the golden certificate gate. Pass-label lists are asserted exactly by `tests/test_render_certificate_contract.py` — add timings without renaming or reordering pass labels.
</context>

<what_to_fix>
Ordered by severity. Audit finding IDs refer to `docs/audits/fable5-moonshots/14-censor-implementation-audit.md` § Findings.

1. **F-02 — Golden negative control must reject even in baseline-update mode.**
   `tests/test_recipe_goldens.py:24` binds `UPDATE_GOLDENS` at import; `_assert_matches_golden` (`:904`) and `_emit_or_verify_certificate` (`:966`) branch on that constant, so the negative control `test_recipe_golden_gate_rejects_pixel_regression` (`:1036-1054`) cannot disable update mode via `monkeypatch.delenv` — under `FORGE3D_UPDATE_RECIPE_GOLDENS=1` it would take the update branch, copy the deliberately corrupted image over the committed golden, and never raise. Fix: make update-mode a call-time decision (read the env inside the functions, or accept an explicit `update=` parameter resolved per call), and make the negative control force it off through that mechanism. Add a regression test that sets the env and proves the control still rejects and the committed golden is untouched.

2. **F-04 — Populate live `gpu_ms` on every GPU-backed certified render path; honest `timestamp_valid`.**
   Real timestamps currently reach certificates only from `src/terrain/renderer/core.rs:436-444` and `src/scene/render_paths/timing.rs:109-117`. All other native GPU render paths hardcode `record_pass(label, 0.0, draws)` even when `timestamp_query` is granted: `src/core/certificate.rs:338` (external captures) and ~20 sites under `src/py_functions/` (e.g. `adjudication.rs:76-77`, `frame.rs:33`, `vector/oit.rs:98-99`, `path_tracing/terrain_reference.rs:200-203`), plus `terrain/spike/render.rs:174,176` and `terrain/renderer/offline.rs:1562`. Thread a `GpuTimingManager` scope (begin/end around the pass, `get_results_blocking()` at end-of-render is acceptable for one-shot offscreen renders) through each GPU-backed path so its certificate passes carry real times when the feature is granted. CPU-side passes (labels like `python.*`, `mapscene.finalize`, `sdf.native_cpu`, `renderer.cpu_triangle`, `smoke.cpu_projection`) legitimately stay 0.0 — do not fabricate. Also fix `src/core/gpu_timing.rs:525`: `timestamp_valid` is unconditionally `true`; derive it from whether real timestamps resolved. Committed certificates' gpu_ms values are unsigned; regenerating them for this is NOT required.

3. **F-01 — Keep golden negative-control semantics without routine scratch corruption.**
   The focused contract must prove that update mode cannot overwrite a baseline during a rejecting comparison and that probe `absent` differs from probe `crash`. Reuse the latest attributable red-proof while golden comparison, probe routing, and aggregate enforcement are unchanged. Generate a new scratch corruption only during explicit acceptance/release after changing one of those mechanisms; never require a downstream feature PR to recreate it.

4. **F-03 — Keep `proj` and `geos-topology` feature truth.**
   Routine CI verifies that declarations, wheel features, and acceptance routing agree and compiles the affected portable surface. The complete feature matrix plus dedicated system-PROJ execution is acceptance/release evidence, not a routine merge requirement.

5. **F-05 — Certify the BRDF tile renders; make the contract test harder to evade.**
   `render_brdf_tile` / `render_brdf_tile_overrides` (`src/py_functions/brdf/wrappers.rs:5-16`, registered in `src/py_module/functions/rendering.rs:11-12`) produce pixel arrays with no `certificate=` kwarg, no capture, no pass record, no shader-use record. Add the standard contract: `certificate=None` kwarg routed through `emit_certificate_for_kwarg` (`src/core/certificate.rs:503-528`), a render capture around the render (`begin_render_capture_with_resources` … `finish_render_capture` — copy the pattern from `src/py_functions/frame.rs`), `record_pass` with a stable label (e.g. `brdf.tile`), `record_shader_use` for its shader module (route module creation through `create_labeled_shader_module` if it is not already), and real gpu_ms per item 2. Update `python/forge3d/__init__.pyi`. In `tests/test_render_certificate_contract.py`: add `Scene.render_rgba`, `Scene.render_png`, `render_debug_pattern_frame`, and both BRDF functions to the enumeration, and add a sweep-style guard that fails when a public callable whose name matches `render_*` (in `forge3d` and `forge3d._forge3d`) is missing both a `certificate` parameter and an entry in an explicit documented-exclusions list. Put `MapPlate.compose/export_png/export_jpeg`, `oidn_denoise`, `atrous_denoise`, `numpy_to_png`/`png_to_numpy`, viewer/widget snapshot paths, and `export_svg`/`export_pdf` in that exclusions list with one-line reasons (composition/filter/IO/interactive/vector — outside CENSOR's render definition per `14-censor.md:55`), and mirror the reason in each one's docstring.

6. **F-07 — Ledger invariant and ownerless allocations.**
   The `debug_assert_eq!` in `AllocationLedger::snapshot()` (`src/core/resource_tracker.rs:363-373`) only checks the ledger against its own atomic and only runs on the budget-error path. Add a debug-build cross-assert that the ledger's host-visible and device-local totals equal the `ResourceRegistry` counters (both are updated by the same wrappers: `resource_tracker.rs:545,581,606`) at render-capture finish (`finish_ledger_capture`), plus a Rust unit test. Make owner-less allocations visible instead of silently excluded from captures (`LedgerEntry.owner_id == None` entries are invisible today): either attribute them to a sentinel owner included in every capture, or record a structured degradation (`allocation_unattributed`) when a capture finishes while ownerless entries were live-allocated during it. Do not break the existing scoping tests (`test_scene_allocations_ignore_unrelated_live_scene` etc.).

7. **F-10 — Probe must not conflate "no adapter" with "renderer crashed".**
   `scripts/terrain_ci_probe.py:155-159` exits 2 on ANY exception, which the golden job maps to ABSENT-and-success. Use distinct exit codes (e.g. 2 = adapter absent/unsafe → ABSENT path; 3 = probe crashed → job must FAIL). Update the golden job in `.github/workflows/ci.yml` to branch on the probe's recorded outcome/output so crash fails the job, and update gate (f)'s source asserts in `tests/test_no_silent_degradation.py:330-347` to lock the new wiring. Strengthen, never weaken, the gate.

8. **F-08 — Unnegotiated device in `extrude_polygon_gpu_py`.**
   `src/vector/api/extrusion.rs:26-36` builds a private `DeviceDescriptor::default()` device, bypassing capability negotiation, the tracked global device, and the budget. Route it through `crate::core::gpu::try_ctx()` (negotiated + tracked). Extend the source-text capability gate in `tests/test_capability_negotiation.py` to also reject `request_device(&wgpu::DeviceDescriptor::default()` patterns in non-test `src/` code.

9. **F-09 / F-11 — Harden the source gates.**
   `tests/test_allocation_gate.py:19`: extend the regex set to catch UFCS (`::create_buffer(`/`::create_texture(`), `create_texture_with_data(`, and line-split calls (scan with a multiline pattern or strip newlines between a `.`/`::` and the call); keep the tracker file itself excluded and keep the non-vacuity self-test green. `tests/test_dead_render_structure_gate.py`: additionally grep all of `src/` for the forbidden symbols `PostFxChain`, `execute_chain`, `postfx_apply_noop`, `PostFxResourcePool`, `RenderBundleManager`, `render_bundles`, and extend the per-frame `create_bind_group` text check from 2 files to every file in `src/viewer/render/main_loop/` minus an explicit allowlist for the cache/lazy sites (`postfx_cache.rs`, `geometry/pass.rs` lazy autogen).

10. **F-12 — Refresh stale internal docs.**
    `.claude/rules/rust-core.md`: `Session(backend=...)` now honors-or-raises (`src/core/session.rs:42-54`); `src/core/postfx/` and `src/core/framegraph.rs` are deleted. `.claude/rules/build-and-ci.md` and `AGENTS.md` must describe the focused routine profile versus explicit acceptance/release matrix, Metal goldens, signing, and physical GPU lanes. Keep the corrected feature inventory and mark P1.2 bloom/PostFxChain guidance as historical.

11. **F-06 — signing-key provenance belongs to protected acceptance/release.**
    Production signing uses `FORGE3D_CERT_SIGNING_KEY` only in a protected acceptance or release workflow. Routine and fork PRs are explicitly untrusted and verify canonical payloads, certificate contracts, and tamper rejection without the production secret.
</what_to_fix>

<action_safety>
Keep changes tightly scoped to the findings above. No new dependencies (signing stays in-tree `ed25519-dalek`/`_ed25519.py`; hashing `sha2`). Never delete or weaken an invariant test, loosen a golden threshold, or let a crash report `ABSENT`. Do not regenerate committed goldens or certificates unless a legitimate acceptance/release change requires it. Do not rename existing certificate pass labels. Leave `python/forge3d/forge3d.pdb` and `docs/dev/` alone in commits you author except where maturin regenerates the pdb.
</action_safety>

<missing_context_gating>
Do not guess repository facts. Every file:line above was verified on 2026-07-10 at commit `45c14c44`; re-verify before editing (the tree may have moved). If a named symbol is absent, locate it by grep before concluding anything. If required context is genuinely unavailable (e.g. CI secrets, runner hardware behavior), state exactly what remains unknown in the final report instead of substituting an assumption.
</missing_context_gating>

<routine_pr_verification>
After each code fix, run the focused checks for the changed surface:
1. `cargo fmt --check`
2. `cargo forge3d-clippy`
3. Build/install the extension when the change crosses the native boundary.
4. `python scripts/ci_pytest_lane.py --profile fast -q --tb=short`
5. Run directly affected Rust/Python behavior tests and certificate/budget probes.
If a check fails, fix the defect — do not adjust the gate.
</routine_pr_verification>

<acceptance_release_verification>
Only for an explicitly requested acceptance/release candidate, additionally run
the curated cross-platform Rust/feature matrix, release native and wheel builds,
`python scripts/ci_pytest_lane.py --profile full`, candidate-selected golden and
physical-GPU probes, production-signature sweep, and red-proof evidence when the
golden-routing mechanism changed.
</acceptance_release_verification>

<completeness_contract>
Resolve every implementation finding before stopping. External CI, physical-GPU, production-signing, exhaustive-golden, and red-proof evidence is required only for an explicitly designated acceptance/release candidate. If routine work changes one of those mechanisms, run focused affected-integration checks and record fresh acceptance evidence as pending; do not promote the pull request into the closure suite. Report missing external evidence separately as `NOT_PROVEN`; do not block an unrelated implementation or weaken a truth invariant.
</completeness_contract>

<structured_output_contract>
Final response, in order:
1. Per finding F-01..F-12: `closed | not closed`, one line of evidence each (command + outcome, or run URL + failing step name for CI items).
2. Focused routine verification outcomes.
3. Acceptance/release evidence only when that lane was explicitly requested.
4. Exact residual implementation blockers and separately labeled acceptance evidence gaps.
Keep it compact; no scene-setting.
</structured_output_contract>
