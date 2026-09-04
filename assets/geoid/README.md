# EGM96 geoid coefficients (degree/order 120)

`egm96_n120.bin` is the committed coefficient set behind
`forge3d::geo::geoid` / `forge3d.crs.geoid_undulation`. It is evaluated by
spherical-harmonic synthesis on demand — it is never expanded into a grid.

## Provenance

Derived from the official NGA/NASA EGM96 release (public data):

- `EGM96` — the NASA/NIMA spherical-harmonic potential coefficient set,
  complete to degree and order 360, from the NGA "egm-96spherical" package
  (`https://earth-info.nga.mil/php/download.php?file=egm-96spherical`),
  truncated here to degree/order 120.
- `CORRCOEF` — NGA's spherical-harmonic correction coefficients (primarily the
  height-anomaly → geoid-undulation conversion term of Rapp 1996), same
  package, truncated to degree/order 120, stored in centimetres as published.

The downloaded `egm-96spherical.zip` archive is 5,011,663 bytes with
SHA-256 `1f21ab8151c1b9fe25f483a4f6b78acdbf5306daf923725017b83d87a5f33472`.
Its original `EGM96` and `CORRCOEF` members respectively hash to
`1e5e6c30343989b8e2eda0bb96bde06ef05981eec69934d6e44aace4f0d6a9d5` and
`72a3dbcf1c5cd60602770e38b5145d09b5ba5c47e72625b2159bdf48139d8b2d`.
Extraction retains n=2..120 for `EGM96` and n=0..120 for `CORRCOEF`, storing
their C/S pairs in n-major order.

The evaluation convention follows NGA's reference program `F477.F` exactly:
WGS84(G873) constants, removal of the normal field's even zonal harmonics
(J2..J10), the correction-model sum, and the −0.53 m zero-degree term, so the
undulations refer to the WGS84 ellipsoid. Reference values for validation are
committed under `tests/data/egm96_test_values.txt` (NGA `OUTF477.DAT` points
plus `WW15MGH.GRD` grid nodes).

## Binary format (little-endian)

| offset | type      | value |
|--------|-----------|-------|
| 0      | `[u8; 8]` | magic `"F3DEGM96"` |
| 8      | `u32`     | format version (1) |
| 12     | `u32`     | nmax (120) |
| 16     | `u32`     | potential pair count (7378) |
| 20     | `u32`     | correction pair count (7381) |
| 24     | `(f64, f64) × 7378` | fully-normalized (C̄nm, S̄nm), n = 2..=120, m = 0..=n, n-major |
| …      | `(f64, f64) × 7381` | correction (Cnm, Snm) in centimetres, n = 0..=120, m = 0..=n, n-major |

Total size: 236,168 bytes (< 1 MiB).

Committed artifact SHA-256:
`b640e9dcefd1040f0b184a101e1eab2740486a85680a560080ec091eab796fe4`.
Run `python scripts/verify_mensura_fixtures.py` to verify the committed
artifacts without network access. Passing an already-downloaded NGA archive to
`--egm96-spherical` verifies the deterministic extraction against its original
members; the script never downloads data.

# Mars GMM3 areoid

`mars_areoid_n179.bin` is the compact harmonic surface behind
`forge3d.crs.areoid_undulation`. Its source is NASA PDS product
`GGMRO_120_GEOID_90.IMG`, an areoid map generated from GMM3 truncated from
degree 3 through 90. The PDS product uses a 3,396,000 m / 1:196.877360
reference ellipsoid and GM 4.2828372854187757e13 m³/s². Before fitting, its
planetocentric radial heights are converted to the registered IAU 2000 Mars
ellipsoid (3,396,190 m / 1:169.894447223612), so `body_info("mars")` and
`areoid_undulation` share one horizontal and vertical reference.

`scripts/build_selene_areoid.py` deterministically transforms the 180×360
float32 PDS map into fully normalized dimensionless disturbing-potential
coefficients. The degree-179/order-90 representation is needed to preserve the
published ellipsoid-relative map on its equiangular latitude grid. Measured
against all 64,800 source cells, its maximum error is 0.735980 m and RMS error
is 0.037742 m. The 30 independently sampled committed reference cells have
maximum error 0.351391 m and RMS error 0.089154 m.

The little-endian container follows the EGM96 header layout: magic
`F3DAREO1`, version 1, maximum degree 179, 16,290 coefficient pairs, zero
correction pairs, then n-major `(f64 C̄nm, f64 S̄nm)` pairs. Total size is
260,664 bytes. Full source URLs, SHA-256 hashes, fit bounds, and artifact hash
are recorded in `mars_areoid_n179.manifest.json`. The Moon intentionally has
no spherical-harmonic gravity surface.
