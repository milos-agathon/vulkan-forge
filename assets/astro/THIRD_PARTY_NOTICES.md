# Third-party notices — `assets/astro/`

forge3d redistributes the files in this directory inside every published wheel
(they are linked into the native module with `include_bytes!`). The notices
required by their upstream licences are reproduced below. See `MANIFEST.toml`
for the per-file provenance, checksums and declared error budgets.

---

## `moon_terms.bin` — lunar periodic terms

**Underlying published source.** The coefficients are tables 47.A and 47.B of
Jean Meeus, *Astronomical Algorithms*, 2nd edition (Willmann-Bell, 1998),
chapter 47 ("Position of the Moon"). Meeus's tables are a published presentation
of the ELP-2000/82 lunar theory; forge3d uses them as astronomical reference
data. This notice does not claim, and the transcription's licence cannot grant,
any right in Meeus's book itself.

**Transcription.** The machine-readable values were taken from the `astronomia`
project's transcription (`src/moonposition.js`), which is distributed under the
MIT licence:

> The MIT License (MIT)
>
> Copyright (c) 2016 Commenthol
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Source: <https://github.com/commenthol/astronomia>

---

## `vsop87d.bin` — VSOP87D planetary theory

Bretagnon, P. & Francou, G., *Planetary theories in rectangular and spherical
variables: VSOP87 solution*, Astron. Astrophys. 202, 309 (1988). Coefficients
retrieved from the IMCCE public ephemeris archive,
<https://ftp.imcce.fr/pub/ephem/planets/vsop87/>. Published astronomical theory
data, distributed by IMCCE for free use.

## `bright_stars.bin` — Yale Bright Star Catalogue

Hoffleit, D. & Warren, W. H. Jr., *The Bright Star Catalogue*, 5th Revised
Edition (1991), CDS catalogue V/50, retrieved through VizieR
(<https://cdsarc.cds.unistra.fr/viz-bin/cat/V/50>). Public-domain astronomical
catalogue data.

## `delta_t_fit.dat`, `leap_seconds.dat` — Earth-orientation time scales

ΔT nodes from JPL Horizons' EOP series (<https://ssd.jpl.nasa.gov/horizons/>);
leap seconds from IERS Bulletin C
(<https://hpiers.obspm.fr/iers/bul/bulc/bulletinc.dat>). NASA/JPL and IERS
public data.

## `moon_albedo.bin` — lunar albedo texture

Downsampled from the NASA Scientific Visualization Studio *CGI Moon Kit* colour
map (LRO WAC/LOLA), <https://svs.gsfc.nasa.gov/4720/>. NASA public data; credit
NASA's Scientific Visualization Studio.
