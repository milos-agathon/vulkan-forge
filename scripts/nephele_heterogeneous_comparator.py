#!/usr/bin/env python3
"""Execute the fixed 100M-sample independent NEPHELE comparator."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

SAMPLES, SEED, MASK64 = 100_000_000, 0x4E455048454C4501, (1 << 64) - 1

def _canonical(v: object) -> bytes:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def _identity(algorithm: str, parameters: dict[str, Any]) -> dict[str, Any]:
    value = {"algorithm": algorithm, "parameters": parameters}
    return {**value, "sha256": hashlib.sha256(_canonical(value)).hexdigest()}

def comparator_cache_key(record: dict[str, Any]) -> str:
    """Input-only cache identity; no result field can select its own cache key."""
    fields = ("algorithm", "transport_representation", "seed_identity", "sample_mapping", "source_revision", "producer_tool", "medium")
    return hashlib.sha256(_canonical({k: record[k] for k in fields})).hexdigest()

def _uniform_batch(start: int, stop: int, dimension: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        value=np.uint64(SEED) ^ (np.arange(start,stop,dtype=np.uint64)*np.uint64(3)+np.uint64(dimension))
        value += np.uint64(0x9E3779B97F4A7C15)
        value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
        value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
        value ^= value >> np.uint64(31)
    return ((value >> np.uint64(11)).astype(np.float64)+0.5)/float(1<<53)

def _represented_density(raw: list[int], shape: tuple[int, int, int]) -> tuple[np.ndarray, str]:
    decoded = np.asarray(raw, dtype=np.uint16).astype(np.float32) / np.float32(65535.0)
    f16 = decoded.astype(np.float16).reshape((shape[2], shape[1], shape[0]))
    digest = hashlib.sha256(f16.reshape(-1).view(np.uint16).astype("<u2", copy=False).tobytes()).hexdigest()
    return f16.astype(np.float32), digest


def _texel_axis(coordinate: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = coordinate * size - 0.5
    lower_i = np.floor(scaled).astype(np.int64)
    fraction = scaled - np.floor(scaled)
    lower = np.clip(lower_i, 0, size - 1)
    upper = np.clip(lower_i + 1, 0, size - 1)
    return lower, upper, fraction

def produce(medium_path: Path, source_revision: str, raw_path: Path, *, samples: int = SAMPLES) -> dict[str, Any]:
    medium = json.loads(medium_path.read_text(encoding="utf-8")); domain = medium.get("domain", {}); transport = medium.get("transport", {})
    shape = tuple(domain.get("grid_shape", ())); bounds_min=domain.get("bounds_min"); bounds_max=domain.get("bounds_max"); raw = medium.get("density_r16"); sigma_a=medium.get("sigma_a"); sigma_s=medium.get("sigma_s"); axis = transport.get("slab_axis"); density_scale=medium.get("density_scale"); density_transport=medium.get("density_transport"); majorant_transport=medium.get("majorant_transport")
    coefficients = [sigma_a, sigma_s]
    if medium.get("schema") != "forge3d.nephele.heterogeneous_medium/2" or set(medium)!={"schema","domain","density_r16","density_transport","majorant_cells","majorant_transport","transport","sigma_a","sigma_s","phase","density_scale"} or len(source_revision)!=40 or any(c not in "0123456789abcdef" for c in source_revision) or len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape) or not isinstance(bounds_min,list) or not isinstance(bounds_max,list) or len(bounds_min)!=3 or len(bounds_max)!=3 or any(not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isfinite(a) or not math.isfinite(b) or a>=b for a,b in zip(bounds_min,bounds_max)) or not isinstance(raw, list) or len(raw) != math.prod(shape) or any(type(n) is not int or not 0 <= n <= 65535 for n in raw) or any(not isinstance(values,list) or len(values)!=3 or any(isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n<0 for n in values) for values in coefficients) or axis not in (0,1,2) or isinstance(density_scale,bool) or not isinstance(density_scale,(int,float)) or not math.isfinite(density_scale) or density_scale<=0 or samples <= 1:
        raise ValueError("invalid heterogeneous slab")
    represented, represented_sha256 = _represented_density(raw, shape)
    expected_density_transport={"schema":"forge3d.nephele.density_transport/1","decode":"unorm16-div-65535-as-f32-then-ieee-f16-rne","storage":"ieee-f16-bits-little-endian","sampling":"normalized-clamp-to-edge-linear-texel-center-uN-minus-0.5","f16_sha256":represented_sha256}
    expected_majorant_transport={"schema":"forge3d.nephele.majorant_transport/1","grid_shape":list(shape),"query":"floor(clamp(unit,0,1)*N)-clamped-to-N-minus-1","construction":"3x3x3-clamped-neighborhood-trilinear-outward-then-f32-extinction-outward"}
    sigma_t_spectrum=[float(a)+float(s) for a,s in zip(sigma_a,sigma_s)]; sigma_t=max(sigma_t_spectrum); extinction_channel=max(range(3),key=sigma_t_spectrum.__getitem__)
    if sigma_t <= 0 or density_transport!=expected_density_transport or majorant_transport!=expected_majorant_transport or transport!={"sigma_t_spectrum":sigma_t_spectrum,"sigma_t_max_channel":sigma_t,"extinction_channel":extinction_channel,"slab_axis":axis}:
        raise ValueError("heterogeneous slab extinction must be positive")
    transverse=[d for d in range(3) if d!=axis]; slab_distance=float(bounds_max[axis]-bounds_min[axis])
    moved=np.moveaxis(represented,2-axis,0); columns=moved.astype(np.float64).mean(axis=0)*slab_distance*float(density_scale); survived=0; digest=hashlib.sha256(); batch=1_000_000
    with raw_path.open("wb") as stream:
        for start in range(0,samples,batch):
            stop=min(start+batch,samples); coordinates=[_uniform_batch(start,stop,dimension) for dimension in (0,1)]; bases=[]; fractions=[]
            uppers=[]
            for dimension,coordinate in zip(transverse,coordinates):
                lower,upper,fraction=_texel_axis(coordinate,shape[dimension]); bases.append(lower); uppers.append(upper); fractions.append(fraction)
            values=np.zeros(stop-start,dtype=np.float64)
            for upper1 in (0,1):
                for upper0 in (0,1):
                    weight=np.where(upper0,fractions[0],1-fractions[0])*np.where(upper1,fractions[1],1-fractions[1])
                    # `columns` retains the two transverse axes in their
                    # original z/y/x order after the slab axis is removed.
                    world_indices={transverse[0]:uppers[0] if upper0 else bases[0],transverse[1]:uppers[1] if upper1 else bases[1]}
                    sampled=columns[tuple(world_indices[dimension] for dimension in reversed(range(3)) if dimension!=axis)]
                    values += weight*sampled
            sample=_uniform_batch(start,stop,2)<np.exp(-float(sigma_t)*values); survived+=int(np.count_nonzero(sample)); packed=np.packbits(sample,bitorder="little").tobytes(); stream.write(packed); digest.update(packed)
    raw_hash=digest.hexdigest()
    medium_hash = hashlib.sha256(medium_path.read_bytes()).hexdigest(); tool = Path(__file__).resolve(); mean=survived/samples; variance=(survived-survived*survived/samples)/(samples-1)
    representation={"density_f16_sha256":represented_sha256,"grid_shape":list(shape),"density_scale":float(density_scale),"sampling":expected_density_transport["sampling"],"slab_integration":"exact-piecewise-linear-texel-center-column-mean"}
    record = {"schema":"forge3d.nephele.heterogeneous_comparator/4","algorithm":"independent-f16-texel-center-column-bernoulli-v3","transport_representation":representation,"extinction_channel":extinction_channel,"sigma_t":sigma_t,"seed_identity":_identity("splitmix64",{"seed_u64":SEED}),"sample_mapping":_identity("indexed-ray-bernoulli-v1",{"dimensions":["u","v","survival"],"bit_order":"lsb0"}),"source_revision":source_revision,"command":f"python scripts/nephele_heterogeneous_comparator.py {medium_path.name} OUTPUT --source-revision {source_revision}","generated_at_utc":datetime.now(timezone.utc).isoformat(),"producer_tool":{"path":"scripts/nephele_heterogeneous_comparator.py","sha256":hashlib.sha256(tool.read_bytes()).hexdigest()},"medium":{"path":medium_path.as_posix(),"sha256":medium_hash},"samples":{"count":samples,"sum":survived,"sum_squares":survived},"statistics":{"mean":mean,"sample_variance":variance,"standard_error":math.sqrt(variance/samples)},"raw_output":{"path":raw_path.name,"sha256":raw_hash,"encoding":"bit-packed-lsb0-bernoulli","samples":samples},"cache_provenance":{"status":"freshly_executed","cache_key_scope":"inputs-only"}}
    record["cache_key"] = comparator_cache_key(record); return record

def produce_rr_evidence(on: Any, off: Any, source_revision: str, raw_path: Path, analytic_oracle: float) -> dict[str, Any]:
    """Persist paired raw RR-on/off contributions for independent verification."""
    on=[float(value) for value in on]; off=[float(value) for value in off]
    if len(on) != len(off) or len(on) < 2 or len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision) or not math.isfinite(analytic_oracle) or not 0 <= analytic_oracle <= 1:
        raise ValueError("invalid Russian-roulette contribution inputs")
    accumulators: dict[str, dict[str, float | int]] = {}
    with raw_path.open("wb") as stream:
        for left, right in zip(on, off):
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (left, right)):
                raise ValueError("RR contributions must be finite and nonnegative")
            stream.write(struct.pack("<dd", float(left), float(right)))
    for name, values in (("on", on), ("off", off)):
        total = math.fsum(values)
        accumulators[name] = {
            "count": len(values),
            "sum": total,
            "sum_squares": math.fsum(value * value for value in values),
            "minimum": min(values),
            "maximum": max(values),
        }
    tool = Path(__file__).resolve()
    return {
        "schema": "forge3d.nephele.russian_roulette_evidence/3",
        "source_revision": source_revision,
        "analytic_oracle": analytic_oracle,
        "producer_tool": {
            "path": "scripts/nephele_heterogeneous_comparator.py",
            "sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        },
        "sample_mapping": _identity("canonical-delta-track-single-scatter-rr-v1", {"columns": ["rr_on", "rr_off"], "rr_trials_per_collision": 4, "encoding": "little-endian-f64"}),
        "raw_output": {
            "path": raw_path.name,
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "encoding": "little-endian-f64-rr-on-off-pairs",
            "pairs": len(on),
        },
        "accumulators": accumulators,
    }

def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("medium",type=Path); p.add_argument("output",type=Path); p.add_argument("--source-revision",required=True); a=p.parse_args(argv)
    try:
        record=produce(a.medium,a.source_revision,a.output.with_suffix(".samples.bin"))
        if record["samples"]["count"] != SAMPLES or record["samples"]["sum"] == 0: raise ValueError("incomplete or zero comparator")
        a.output.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    except (OSError,ValueError,json.JSONDecodeError) as exc: print(f"NEPHELE comparator failed: {exc}",file=sys.stderr); return 2
    return 0

if __name__ == "__main__": raise SystemExit(main())
