# SPA reference-vector provenance

All 24 rows in `spa_reference.csv` are independently traceable to the official
NLR/NREL Solar Position Algorithm.

- The row-1 zenith/azimuth pair is also the worked example in Reda and Andreas,
  *Solar Position Algorithm for Solar Radiation Applications*,
  NREL/TP-560-34302, revised January 2008, Appendix A
  ([official report](https://www.nrel.gov/docs/fy08osti/34302.pdf)). The paper
  reports equation of time as `14.641503` minutes.
- All CSV output fields, including row 1's auxiliary radius-vector and
  equation-of-time fields, were queried from NLR's
  [public SPA calculator](https://midcdmz.nlr.gov/solpos/spa.html) on
  2026-08-11. The calculator page states that it uses NLR's official SPA and
  submits to `https://midcdmz.nlr.gov/apps/spcalc.exe`. No restricted SPA source
  code was downloaded or redistributed. For the row-1 inputs the calculator
  reports equation of time as `14.641511` minutes; the CSV deliberately records
  that calculator result rather than attributing the auxiliary field to the
  paper.

For every row, the GET request used the CSV's date, civil time, timezone,
latitude, longitude, elevation, pressure, temperature, and delta-T values,
plus these fixed inputs:

```text
algorithm=0
otype=1
dut1=0
azmrot=180
slope=0
refract=0.5667
field=0&field=1&field=14&field=38&field=40&field=41
```

The requested fields map respectively to topocentric zenith, topocentric
azimuth eastward from north, Earth radius vector, uncorrected topocentric
elevation, corrected topocentric elevation, and equation of time. The CSV keeps
the calculator's six-decimal output verbatim for all rows; tests therefore use
the required `0.0003 deg` angular gate and a `5.1e-7` rounding allowance only
for the calculator's six-decimal AU and equation-of-time fields.

Example request (row 1, line-wrapped for readability):

```text
curl --get https://midcdmz.nlr.gov/apps/spcalc.exe \
  --data-urlencode algorithm=0 --data-urlencode syear=2003 \
  --data-urlencode smonth=10 --data-urlencode sday=17 \
  --data-urlencode eyear=2003 --data-urlencode emonth=10 \
  --data-urlencode eday=17 --data-urlencode otype=1 \
  --data-urlencode hr=12 --data-urlencode min=30 --data-urlencode sec=30 \
  --data-urlencode latitude=39.742476 --data-urlencode longitude=-105.1786 \
  --data-urlencode timezone=-7 --data-urlencode elev=1830.14 \
  --data-urlencode press=820 --data-urlencode temp=11 \
  --data-urlencode dut1=0 --data-urlencode deltat=67 \
  --data-urlencode azmrot=180 --data-urlencode slope=0 \
  --data-urlencode refract=0.5667 \
  --data-urlencode field=0 --data-urlencode field=1 \
  --data-urlencode field=14 --data-urlencode field=38 \
  --data-urlencode field=40 --data-urlencode field=41
```
