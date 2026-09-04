# Virtual Texturing Support Matrix

`MapScene.validate()` reports unsupported virtual-texture configurations before
rendering. Unknown layer families fail with `vt_unsupported_family`; the
runtime is not an albedo-only path—albedo, normal, and mask families share the
same store, feedback, residency, page-table, and atlas machinery.

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Albedo terrain VT family | `supported` | Runtime pages BC7 albedo through the shared store, feedback, residency, page-table, and atlas path. | `terrain_vt_bc_atlas` / `terrain_vt_bindless_atlas` certificate degradations when the compressed bindless path is unavailable. |
| Normal terrain VT family | `supported` | Runtime pages BC5 normal data through the same family-aware path as albedo. | Same explicit compressed/bindless atlas degradations; no silent family skip. |
| Mask terrain VT family | `supported` | Runtime pages BC7 mask-roughness-metalness data through the same family-aware path as albedo. | Same explicit compressed/bindless atlas degradations; no silent family skip. |
| Runtime residency and footprint stats | `supported` | `vt_stats()` reports aggregate and per-family residency budgets plus physical compressed-atlas and uncompressed-equivalent capacities. | `unavailable_cache_lod_stats` remains the product-level diagnostic when a higher-level integration cannot obtain runtime stats. |

Unknown material families must not silently skip: the typed Python layer rejects
them, while the lower-level compatibility registration logs a warning and keeps
the source for diagnostics without enabling it. The supported material-family
set is exactly `albedo`, `normal`, and `mask`.
Height streaming is a separate store-backed height mosaic, not a fourth slot in
the material-family atlas or its per-family residency split.

## Memory budgets

The terrain virtual-texture runtime is governed by **two independent budgets**.
They are enforced separately and are not interchangeable.

| Budget | Value | Enforcement |
| --- | --- | --- |
| Host-visible (staging rings, upload buffers, readbacks) | **512 MiB** | CENSOR's resource tracker, enforce-by-default (`BUDGET_POLICY_ENFORCE`); an over-budget host-visible allocation fails with `RenderError::Budget` |
| Device-local VT atlas (the resident tile atlas itself) | **256 MiB by default**, set by `TerrainVTSettings.residency_budget_mb` | Per-family LRU eviction in `FamilyResidencyTracker`, with the shared tile-cache capacity as the global backstop |
| Per-frame VT upload (bytes copied into the atlas per frame) | Caller-supplied, `vt_upload_budget_bytes` | Highest-priority-first: a request whose upload would push the frame over the budget is skipped and stays in the retained request set for a later frame |

The device-local atlas budget is deliberately **not** counted against the
512 MiB host-visible budget: atlas tiles are device-local, so they never
occupy the host-visible heap that CENSOR's tracker governs. Compressed
(BC7/BC5) atlas capacity is reported against each family's native raw format:
albedo and mask are 4:1 (RGBA8 to BC7), and normal is 2:1 (RG8 to BC5).
With all three material families enabled, the combined raw-equivalent to
compressed capacity ratio is therefore 10:3 (about 3.33:1).

`residency_budget_mb` is split **evenly across the enabled families**
(albedo, normal, mask). With all three enabled at the 256 MiB default each
family gets one third of the integer byte budget, and one family's paging pressure evicts only its own
least-recently-used tiles while it stays inside that share.

Observed values are reported by `forge3d.diagnostics.vt_stats()`:
`cache_budget_mb`, `atlas_device_local_bytes`,
`atlas_uncompressed_equivalent_bytes`, `atlas_compression_ratio`,
`atlas_device_local_bytes_{albedo,normal,mask}`,
`atlas_uncompressed_equivalent_bytes_{albedo,normal,mask}`,
`atlas_compression_ratio_{albedo,normal,mask}`,
`evictions`, `tiles_streamed`, and `retained_requests`.

On the compatibility path the three material families dynamically share one
RGBA8 atlas, so the aggregate physical/uncompressed ratio is exactly 1:1 and
the per-family footprint fields are `0`: assigning portions of a shared
physical texture to individual families would be invented evidence.
