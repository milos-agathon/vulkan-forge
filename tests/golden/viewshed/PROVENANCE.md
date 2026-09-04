# HELIOS WhiteboxTools viewshed reference

The committed reference is a controlled geodetic workload generated offline
with WhiteboxTools 2.4.0 `Viewshed`. It deliberately uses a zero-height DEM:
that removes rugged-raster interpolation differences between Whitebox's
horizon algorithm and HELIOS' continuous bilinear traversal while keeping the
ellipsoidal curvature/refraction correction load-bearing. The separate Swiss
real-DEM test remains the required >=60 km physical anti-no-op workload.

- Geodetic grid: 256 x 256 cell-centre samples over EPSG:4326 bounds
  `(-0.5, -0.5, 0.5, 0.5)`, all ellipsoidal heights 0 m.
- Observer: raster cell `(row=127, column=127)`, WGS84 cell centre
  `(lat=0.001953125, lon=-0.001953125)`, 250 m AGL.
- Whitebox grid: observer-centred WGS84 azimuthal equidistant CRS. Its affine
  x/y cell sizes are independently calculated by PyProj 3.7.2 `Geod(WGS84)`
  for one longitude/latitude raster step at the observer. The station is
  `(0, 0)` and each DEM value remains tied to the corresponding HELIOS cell.
- EffectiveRadius physics: before Whitebox, each cell is replaced by
  `h_eff = h - d^2(1-k)/(2R_local)` for `k=0.13`. PyProj supplies WGS84
  observer-to-cell geodesic distance and azimuth. `R_local` uses the WGS84
  meridional and prime-vertical radii with Euler's azimuthal formula.
  Whitebox independently solves the projected LOS/horizon problem on that
  effective terrain; Whitebox itself is not claimed to model curvature.
- Flat negative: the identical zero-height DEM and Whitebox command are used
  without the effective-height correction.

Committed outputs:

- `whitebox_curved_analytic_256.png`: SHA-256
  `b40d22f15fafd71965382cdbf0a7b544069542e7ea5a510488a803828163055d`,
  57,957 visible pixels.
- `whitebox_flat_analytic_256.png`: SHA-256
  `cb11c310d80215f00aad6d08e30bfb00e4ac0849dc5314722bd4636cf4611622`,
  65,536 visible pixels.
- Discrimination: Flat-vs-EffectiveRadius IoU is exactly
  `0.8843536376953125`, with 7,579 flipped pixels. The Flat negative therefore
  fails the same 0.98 IoU gate the curved implementation must pass.
- Independent analytic cross-check: evaluating the convex curved sightline
  over the zero terrain gives 57,813 visible pixels and matches Whitebox's
  curved mask at IoU `0.9975153993477923` (144 boundary pixels differ).

The official Apple-Silicon distribution was downloaded from
`https://www.whiteboxgeo.com/WBT_Darwin/WhiteboxTools_darwin_m_series.zip`:

- archive SHA-256: `b0ff9ad48769df68a604c266e249eb4867ce7505fc297f439cf2cefe0c185c8b`
- `WBT/whitebox_tools` SHA-256:
  `63ce77174e2abc32590df2b4868b6fb4123d8598a540290ef9669b2cb919647b`
- `whitebox_tools --version`: `WhiteboxTools v2.4.0`

Install the offline-only dependency `pyshp==3.1.6`, then regenerate with:

```text
python tests/golden/viewshed/generate_reference.py --whitebox-tools /path/to/WBT/whitebox_tools --earth-model effective-radius --observer-height 250 --observer-row 127 --observer-column 127
python tests/golden/viewshed/generate_reference.py --whitebox-tools /path/to/WBT/whitebox_tools --earth-model flat --observer-height 250 --observer-row 127 --observer-column 127
```

The generator prints and executes WhiteboxTools with `--run=Viewshed`,
`--dem=dem.tif`, `--stations=station.shp`, `--output=whitebox.tif`,
`--height=250.0`, and `--compress_rasters=False`. Values greater than zero
are mapped to opaque white and all other values to opaque black.
