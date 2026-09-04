# CENSOR Final Implementation Audit

**Audit date:** 2026-07-11
**Authoritative branch:** `codex/censor-final`
**Audited implementation tree:** `ec2487c6` plus this documentation-only audit commit
**Source requirements:** `14-censor.md`, `14-censor-remediation.md`, and AGENTS.md CENSOR entries

**Validation-policy update (2026-08-05):** `docs/censor-validation-policy.md`
separates implementation truth from acceptance/release evidence. The historical
full-matrix, signing, golden, hardware, and red-proof results below apply only
to their audited SHA and are not routine pull-request requirements.

## Verdict

The implementation inventory is 20 full / 0 partial / 0 none. CENSOR's runtime
capability, degradation, allocation, budget, certificate, shader, and probe
truth mechanisms are complete. Exact-head full-matrix, production-signing,
all-golden, physical-GPU, and scratch-red-proof results are separate
acceptance/release evidence and do not block unrelated pull requests.

## F-01 through F-12

| Finding | Status | Implementation and fresh verification |
| --- | --- | --- |
| F-01 exact-head green/red proof | historical acceptance evidence | Workflow distinguishes `ran`, `absent`, and crash; prior mechanism proof: green run [29130321569](https://github.com/milos-agathon/forge3d/actions/runs/29130321569), red run [29130343077](https://github.com/milos-agathon/forge3d/actions/runs/29130343077). Reuse this evidence while routing is unchanged; produce a new scratch proof only for explicit acceptance/release after changing the mechanism. |
| F-02 update-safe negative control | full | Update mode is read at call time. Ordinary and update-mode invocations pass; baseline SHA remained `DFB429DE…15C6C` before/after. |
| F-03 executed feature coverage | implementation full; acceptance historical | The audited 2026-07-10 snapshot used 15 features. The current inventory adds `shader-contract-asserts`; declarations and lane routing are locked separately from execution. A manual `scope=full` run installs PROJ prerequisites and executes the dedicated `proj` check. Full-matrix logs are acceptance/release evidence, not a routine merge prerequisite. |
| F-04 honest GPU timing | full | Every certified GPU surface uses `OneShotTiming` or its renderer timing manager. Valid stamps produce live values; invalid stamps produce zero. Setup, empty resolution, and readback failure record render-local `timing_unavailable` degradations; the forced-failure/no-leak unit test passes. |
| F-05 render-surface contract | full | BRDF functions and all discovered public `render_*` callables accept `certificate=` or have docstring-mirrored exclusions. Focused contract suite passes. |
| F-06 production signing provenance | full for protected acceptance/release | A random 32-byte seed is provisioned only as the Actions secret `FORGE3D_CERT_SIGNING_KEY`; only its public key is tracked. Protected acceptance/release lanes fail without the secret or on any mismatch. Routine and fork PRs are explicitly untrusted and verify canonicalization, contract shape, and tamper rejection without requiring the secret. |
| F-07 ledger/registry invariant | full | Capture finish compares both ledger axes for equality with `ResourceRegistry::ledger_totals()`, the exact wrapper-owned subset. Ownerless mid-capture allocations are included; mismatch and locality tests pass. |
| F-08 negotiated device gate | full | Extrusion uses `try_ctx()` and the source gate rejects default/qualified/line-split empty-feature device requests. |
| F-09 allocation gate | full | Gate detects instance, UFCS, buffer-init, texture-with-data, and line-split creation. Tracker is the sole exception; non-vacuity tests pass. |
| F-10 probe outcomes | full | Exit 0 is positive, 2 is genuine absence, and 3 is renderer crash. Only 2 writes ABSENT; no `continue-on-error` masks the probe or comparison. |
| F-11 dead structure/per-frame allocation | full | Deleted structures are scanned repo-wide. Viewer sky/fog/postfx bind groups are cached; resize/IBL resource changes invalidate the relevant caches. Routine frame files contain no bind-group or sampler creation. |
| F-12 tracked documentation | full | AGENTS, `.claude` Rust/CI rules, this audit, feature lists, backend semantics, budget default, and deleted structures are tracked and consistent. The audit has a narrow `.gitignore` exception. |

## Requirement inventory

| ID | Requirement | Status |
| --- | --- | --- |
| CENSOR-01 | Capability negotiation | full |
| CENSOR-02 | Capability degradation recording | full |
| CENSOR-03 | Live and failure-honest GPU timings | full |
| CENSOR-04 | Device/pipeline error propagation | full |
| CENSOR-05 | Honest `Session(backend=...)` | full |
| CENSOR-06 | Total tracked allocation surface | full |
| CENSOR-07 | Enforce-by-default budget | full |
| CENSOR-08 | Render-local ledger equality | full |
| CENSOR-09 | Duplicate budget removal | full |
| CENSOR-10 | Complete certificate render surface | full |
| CENSOR-11 | Deterministic production signatures | full |
| CENSOR-12 | Normalized executed-WGSL provenance | full |
| CENSOR-13 | Single render-local degradation sink | full |
| CENSOR-14 | Offline verifier without native module | full |
| CENSOR-15 | Honesty gates and expiring accounting | full |
| CENSOR-16 | Cargo/wheel/PROJ feature truth | full |
| CENSOR-17 | Golden pass/fail/absent semantics | full |
| CENSOR-18 | Dead-structure and allocation closure | full |
| CENSOR-19 | Seven implementation-level measurable wins | full |
| CENSOR-20 | Acceptance/release matrix, signing, golden, hardware, and red-proof evidence | full for audited SHA; rerun only for explicit acceptance/release |

**Counts:** full 20, partial 0, none 0.

## Historical acceptance snapshot

These results describe the audited SHA. They are evidence that the mechanism
worked at closure, not a mandatory routine-PR command list.

| Command/gate | Result |
| --- | --- |
| `cargo fmt --check` | exit 0 |
| `cargo forge3d-clippy` | exit 0 |
| `maturin develop --release` | exit 0 |
| `maturin build --release` | exit 0 |
| Historical curated 15-feature `cargo test` | exit 0; 692 library tests discovered; current inventory also contains `shader-contract-asserts` |
| Focused CENSOR pytest | 89 passed |
| Update-mode negative control | 2 passed; baseline SHA identical before/after |
| Full clean-checkout Python lane | 2709 passed, 139 skipped |
| Local Vulkan terrain goldens | probe positive; 19 passed |
| Local recipe goldens | 27 passed |
| WGSL normalization/mutation unit | LF == CRLF hash; real byte mutation changes hash |
| Production certificate sweep | 22/22 verify; dev public key differs |

The original dirty checkout was never staged, reset, stashed, or edited by this
work. Its geodesy changes and generated PDB are excluded. Backup refs preserve
both superseded CENSOR tips. No red-proof corruption is present in this tree.
