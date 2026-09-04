<p align="center">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/logo/forge3d_dark.svg" alt="forge3d" width="300">
</p>

<p align="center">
  <strong>Path-traced terrain and cartography for Python.</strong><br>
  Rust + WebGPU underneath. Real elevation data in, print-resolution maps out.
</p>

<p align="center">
  <a href="https://pypi.org/project/forge3d/"><img src="https://img.shields.io/pypi/v/forge3d?color=EFA026&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/forge3d/"><img src="https://img.shields.io/pypi/pyversions/forge3d?color=D0C8BA&style=flat-square" alt="Python 3.10+"></a>
  <a href="https://github.com/milos-agathon/forge3d/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://milos-agathon.github.io/forge3d/"><img src="https://img.shields.io/badge/docs-online-blue?style=flat-square" alt="Docs"></a>
</p>

<p align="center">
  <a href="https://milosgis.com/">
    <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/shasta-hero.webp" alt="Mount Shasta rendered by forge3d, snow and glaciers picked out by low-angle sun against a dark sky" width="860">
  </a>
</p>

<p align="center">
  <sub><b>Mount Shasta</b>, path-traced with global illumination and terrain shadows.<br>
  <a href="https://milosgis.com/"><b>Drag to orbit it live &rsaquo;</b></a> &nbsp;·&nbsp; USGS 3DEP elevation &nbsp;·&nbsp; 23 cloud-free Sentinel-2 composites</sub>
</p>

---

## Install

```bash
pip install forge3d
```

<details>
<summary>Optional extras</summary>

```bash
pip install "forge3d[jupyter]"   # notebook widget support
pip install "forge3d[datasets]"  # on-demand sample datasets
pip install "forge3d[all]"       # everything
```

</details>

## Sixty seconds to a render

```python
import forge3d as f3d

dem_path = f3d.fetch_dem("rainier")

with f3d.open_viewer_async(terrain_path=dem_path, width=1440, height=900) as viewer:
    viewer.set_z_scale(0.1)
    viewer.set_orbit_camera(phi_deg=28, theta_deg=49, radius=5400, fov_deg=42)
    viewer.set_sun(azimuth_deg=302, elevation_deg=24)
    viewer.snapshot("rainier.png", width=1920, height=1080)
```

`fetch_dem` pulls the elevation model, the viewer opens a real window you can orbit,
and `snapshot` writes the frame at whatever resolution you ask for.

---

## Made with forge3d

Every map below is rendered end to end by forge3d — titles, legends and scale bars
included. Nothing here was hand-edited or composited in another tool.

<table>
<tr>
<td align="center" width="33%">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/lyon.webp" alt="Circular plan view of Lyon with every building extruded, the Rhone and Saone running through it" width="270"><br>
  <sub><b>Lyon, LOD2 buildings</b><br>CityGML extrusion over RGE ALTI terrain</sub>
</td>
<td align="center" width="33%">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/egypt.webp" alt="Egypt in cold blue relief with the Nile and its delta glowing orange" width="270"><br>
  <sub><b>Egypt, population</b><br>GHSL R2023A density over shaded relief</sub>
</td>
<td align="center" width="33%">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/france.webp" alt="France coloured by a two-variable temperature and precipitation scheme" width="270"><br>
  <sub><b>France, climate</b><br>TerraClimate temperature × precipitation</sub>
</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="50%">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/iberia.webp" alt="Iberian peninsula land cover, forest greens along the north and mountain ranges, crop yellows across the meseta" width="415"><br>
  <sub><b>Iberia, land cover 2024</b> — Sentinel-2 at 10 m classified over terrain, legend composed in-engine</sub>
</td>
<td align="center" width="50%">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/germany.webp" alt="Germany in hypsometric relief with the Rhine, Elbe and Danube drainage networks traced in blue" width="415"><br>
  <sub><b>Germany, hydrology</b> — HydroSHEDS rivers and lakes over a hypsometric DEM</sub>
</td>
</tr>
</table>

<p align="center">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/turkiye.webp" alt="Turkiye seen at a low angle with population density extruded as thousands of spikes, Istanbul the tallest" width="850">
</p>

<p align="center">
  <sub><b>Türkiye, population density</b> — a million WorldPop 2020 cells extruded on a log scale, labelled and lit in a single pass</sub>
</p>

<table>
<tr>
<td width="34%">
  <img src="https://raw.githubusercontent.com/milos-agathon/forge3d/main/docs/assets/readme/california-smoke.webp" alt="Animated loop of wildfire smoke plumes drifting across California through summer 2025" width="280">
</td>
<td width="66%">

**And it moves.**

Volumetric smoke driven by HRRR forecast fields, advected across California
through the summer 2025 fire season and rendered frame by frame — same
scene graph as the stills, one camera, no compositing.

Scenes are driven from Python end to end, so a time axis is just another loop:
fetch the fields, step the clock, call `snapshot`, hand the frames to ffmpeg.

</td>
</tr>
</table>

<sub>The renders above are downsampled for this page. Full-resolution masters — up to 7200×7200 — are on the <a href="https://milos-agathon.github.io/forge3d/gallery/index.html">gallery</a>.</sub>

---

## What forge3d covers

| | |
|---|---|
| **Terrain** | Interactive viewing via `open_viewer_async()` and `ViewerHandle`; snapshots from GeoTIFF or `numpy` DEMs; clipmaps for large regions |
| **Rendering** | Path tracing, PBR materials, subsurface scattering, water and reflections, cloud and contact shadows, AOV and EXR output |
| **Data** | COG streaming, CRS helpers, LAZ/COPC/EPT point clouds, 3D Tiles, GeoJSON, CityJSON, on-demand sample datasets |
| **Cartography** | Raster and vector overlays, labels with halo and occlusion, graticules, `Legend`, `ScaleBar`, `NorthArrow`, `MapPlate` |
| **Offscreen** | `Scene`, `Session`, `TerrainRenderer` and `TerrainRenderParams` for headless and batch work |
| **Output** | PNG and PNG16, SVG and PDF vector export, scene bundles, notebook widgets |

`MapPlate` composition, vector export and the building import pipelines are Pro
features; set a key with `forge3d.set_license_key(...)` to unlock them.

## Documentation

- [**Quickstart**](https://milos-agathon.github.io/forge3d/start/quickstart.html) — install, first viewer session, first overlay, first notebook widget
- [**Feature Map**](https://milos-agathon.github.io/forge3d/guides/feature_map.html) — repo-wide overview of the supported workflows
- [**Tutorials**](https://milos-agathon.github.io/forge3d/tutorials/index.html) — guided GIS and Python tracks
- [**Gallery**](https://milos-agathon.github.io/forge3d/gallery/index.html) — finished recipes at full resolution
- [**API Reference**](https://milos-agathon.github.io/forge3d/api/api_reference.html) — the full public Python surface

## License

The open-source core is released under Apache-2.0 OR MIT. Pro-gated features
require a commercial license key.
