# CI validation tiers

forge3d separates merge validation from high-cost acceptance. A green hosted
check is evidence only for the claim named by that check; it is not evidence of
physical NVIDIA/Vulkan execution.

## Pull-request core

`PR Core Success` is the stable branch-protection context for every pull
request to `main` or `develop`. It requires:

- the cheap workflow-contract preflight before fan-out;
- one exact-head fast contract: formatting, the focused routine clippy surface,
  a source-level SIMD guard, one local extension build, and the deliberately
  small `scripts/ci_pytest_lane.py --profile fast` contract suite.

It does not require the complete Rust/Python/platform matrix, production
signing, all-golden rendering, hosted determinism, a self-hosted physical GPU,
the long fuzz/slow lanes, or deliberate red-proof corruption. Those remain
available as acceptance evidence without becoming a constitution for ordinary
feature work.

For pull requests, the preflight checks out the live base branch, fetches and
verifies the immutable PR head, and materializes their merge locally before it
runs the workflow contracts from a separate base-owned checkout. This avoids
stale activity-event SHAs, makes policy tests added on the base branch available
to older PR heads, and prevents a candidate from weakening its own validator. A
merge conflict fails preflight. The fast contract is built from the exact PR
head SHA.

The larger hosted suites, independently built platform wheels, visual goldens,
and physical evidence run only on the nightly schedule or an explicit manual
scope. A manual `core` scope is available when hosted core/compatibility
evidence is useful without selecting physical acceptance; `full` selects the
complete acceptance set.

Configure branch protection to require `PR Core Success`. Do not require `Full
Acceptance Summary` globally: it is intentionally absent from routine pull
requests and ordinary pushes.

After this policy first lands on the base branch, each already-open pull
request needs one new `pull_request` event so it receives the new required
context. Use **Update branch**, rebase/merge the current base branch into the
PR, or push a new commit. Future PRs receive the check automatically.

## Hosted and exhaustive acceptance

The complete Rust feature matrix, cross-platform wheels, full Python profiles,
slow/fuzz lanes, documentation, and all-golden render lane are acceptance
evidence. They run on the nightly schedule or an explicit manual `full` scope,
not on pull requests or ordinary pushes. Manual `core` selects the hosted
core/compatibility subset when requested.

The render, F3DZ, and ANAMNESIS determinism families install the exact Linux,
Windows, or macOS wheel artifacts built in the caller run; they do not rebuild
the extension. A nightly run and manual `full` or `determinism` scope select
the relevant complete set. They are never path-selected PR gates.

Hosted render legs may record an explicit `ABSENT` result when no qualifying
adapter exists. That result documents hosted-runner capability only and never
satisfies a physical acceptance requirement.

## Physical acceptance

M-06, F3DZ, ANAMNESIS, and TESSELLA keep their existing exact NVIDIA/Vulkan or
DX12 acceptance commands, zero-skip contracts, thresholds, and 90-day evidence.
The frequency is narrow:

- the nightly schedule runs every physical family;
- manual `full`, `m06`, `f3dz`, `anamnesis`, or `tessella` selects the named
  family;
- pull requests and ordinary pushes never allocate a self-hosted GPU runner.

`Full Acceptance Summary` validates every selected hosted and physical family.
It is a reporting/acceptance context, not the global merge gate.

## Certificate refresh

Certificate mutation is not part of CI. `Refresh recipe certificates` is a
manual-only workflow that runs only from the protected `main` ref, builds one
Windows wheel for that exact SHA, and proves the checkout on the physical
NVIDIA/Vulkan runner. Its signing job uses the protected `production-signing`
Environment, requires the production signing secret, renders/signs the catalog,
and uploads the signed certificates plus provenance for 90 days. Configure the
secret and required reviewer on that Environment before enabling the workflow.
Routine PRs verify certificate structure, canonicalization, the pinned public
key, and tamper rejection without possessing or exercising the production key.
