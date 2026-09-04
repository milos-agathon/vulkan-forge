# CENSOR Closure Design

**Date:** 2026-07-10
**Source:** `docs/prompts/fable5-moonshots/14-censor.md`
**Goal:** Close every requirement and every defect found by the 2026-07-10 implementation audit, then produce an independent Fable 5 audit prompt.

> **Current validation policy (2026-08-05):** This document records the
> historical closure design. `docs/censor-validation-policy.md` supersedes its
> routine-CI assumptions. The runtime truth mechanisms remain authoritative;
> full matrices, production signing, all-golden execution, physical-GPU proof,
> and scratch red proof are explicit acceptance/release evidence. Current CI
> reports routine branch protection through `PR Core Success` and named
> acceptance evidence through `Full Acceptance Summary`; a complete candidate
> run is selected with `scope=full`.

## Decisions

Work proceeds as bounded, test-first slices. Each slice must leave a runnable, reviewable repository state and must prove the relevant CENSOR requirements directly. Existing helpers are extended or deleted before any new abstraction is considered. No dependencies are added.

### 1. Render-local certificate state

The certificate must describe one render, not process lifetime. `begin_render_capture` will establish render-local allocation and degradation state; `finish_render_capture` will freeze only values observed in that capture. Consecutive identical renders on the same adapter must therefore have identical signed payloads even when earlier tests or renders allocated resources.

Shader hashes must be tied to the render capture rather than returned from an ever-growing process-global registry. The existing shader creation helper remains the source of hashes over preprocessed WGSL. Render paths will expose only the hashes belonging to the renderer/pass set used by that capture.

Canonical JSON will have one shared pure-Python implementation extracted from the existing recipe-manifest encoder. It will normalize negative zero and reject non-finite floats. Signing will cross the existing native boundary and use the repository's `ed25519-dalek`; the standalone verifier remains pure Python and importable without the native extension.

### 2. GPU truthfulness

Every production device request will route through the existing capability negotiation policy; production `Features::empty()` requests are removed. Pipeline creation will use the existing validation-scope helper, enforced by a source test covering every render and compute pipeline constructor. Device errors remain surfaced as structured render errors/degradations.

Interactive timing will poll without blocking and publish completed data from an earlier frame. Offline/reference rendering may retain an explicit blocking resolve because it is not on an interactive frame loop.

### 3. Allocation truthfulness

The existing tracked allocation wrappers remain the only allocation boundary. Texture byte accounting will use the whole descriptor: block dimensions, mip extents, array/depth layers, and sample count. The ledger will expose render-local peaks and maintain its existing sum invariant. No parallel budget or allocation system will be introduced.

### 4. One degradation and render contract

All actual fallback, placeholder, synthetic, and unsupported-success paths will write to the existing degradation sink. Shared fallback helpers will be fixed once and caller paths audited. Source and behavior tests will cover the named paths in CENSOR and prevent unrecorded siblings.

Every public render entry point will accept the documented `certificate=False|True|path` contract or will explicitly raise `DegradedCapability` when a documented native capability is unavailable. PyO3 registration, package exports, stubs, and API contract locks change together.

### 5. Delete dead architecture

The zero-caller legacy `PostFxChain` and `FrameGraph` surfaces will be deleted rather than routed into live rendering. Remaining viewer post-processing will cache bind groups with the resources that determine them, rebuilding only when those resources change. Every surviving pass remains represented in certificate pass records.

### 6. Literal repository truth gates

Feature declarations, wheel coverage, and acceptance-matrix routing will equal the live referenced Cargo surface; discrepancies cannot hide behind an exclusion constant. Every filesystem `tests/test_*.py` file remains tracked and assigned to the full acceptance profile, an explicit lane, or one current `UNRUN.toml` entry. Routine pull requests run the focused CENSOR truth-contract profile rather than the entire repository suite.

Routine checks lock golden comparison, update-mode safety, and `passed`/`absent`/`failed` routing without modifying a committed baseline. A probe-positive scratch red proof is generated only for explicit acceptance/release when the comparison, probe, or aggregate mechanism changes, and the corruption is then reverted.

### 7. Final evidence and audit prompt

Implementation completion requires focused proof of each changed invariant: format/lint, affected build, capability/degradation/allocation/budget contracts, certificate schema and tamper rejection, the pure offline verifier, and validation-routing tests. Acceptance/release additionally requires the complete Python profile, curated Rust matrix, release native build, candidate-selected terrain/recipe goldens, physical-GPU probes, a production-key certificate sweep, and red-proof evidence when routing changed.

The final deliverable `docs/prompts/fable5-moonshots/14-censor-audit.md` will be based on `docs/fable-5-p0-p1-blender-plan-implementation-audit-prompt.md`, adapted to audit CENSOR's seven measurable wins, public surfaces, CI proof, and current repository state without trusting implementation intent.

## Slice order

1. Certificate determinism, canonicalization, native signing, and exact capture state.
2. Device negotiation, pipeline validation scopes, and non-blocking interactive timing.
3. Descriptor-accurate allocation accounting and render-local ledger evidence.
4. Degradation sink coverage and certificate kwargs on every render entry point.
5. Dead-structure deletion and bind-group lifetime correction.
6. Routine feature/test-accounting truth and golden-routing contracts.
7. Explicit acceptance/release evidence and the Fable 5 audit prompt.

Each slice starts with a failing test, implements the smallest shared/root-cause fix, runs focused verification, and receives a source/diff review before the next slice.
