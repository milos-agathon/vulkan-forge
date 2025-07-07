from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .mesh import Mesh


def _parse_vertex(token: str) -> Tuple[int, Optional[int], Optional[int]]:
    parts = token.split("/")
    vi = int(parts[0]) if parts[0] else 0
    ti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    ni = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    return vi, (ti or None), (ni or None)


def load_obj(path: Union[str, Path]) -> Mesh:
    """Load a minimal OBJ mesh."""
    path = Path(path)
    vertices: List[Tuple[float, float, float]] = []
    normals: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    faces: List[List[Tuple[int, Optional[int], Optional[int]]]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "vn" and len(parts) >= 4:
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "vt" and len(parts) >= 3:
                uvs.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "f":
                face = [_parse_vertex(tok) for tok in parts[1:]]
                faces.append(face)

    vert_map: Dict[Tuple[int, Optional[int], Optional[int]], int] = {}
    pos_out: List[Tuple[float, float, float]] = []
    norm_out: List[Tuple[float, float, float]] = []
    uv_out: List[Tuple[float, float]] = []
    indices: List[int] = []

    for face in faces:
        tris = (
            [face[0], face[1], face[2], face[0], face[2], face[3]]
            if len(face) == 4
            else face
        )
        if len(face) == 4:
            tris = [tris[:3], tris[3:]]
        else:
            tris = [face]
        for tri in tris:
            for vi, ti, ni in tri:
                key = (vi, ti, ni)
                if key not in vert_map:
                    vert_map[key] = len(pos_out)
                    pos_out.append(vertices[vi - 1])
                    if ni is not None and 0 < ni <= len(normals):
                        norm_out.append(normals[ni - 1])
                    if ti is not None and 0 < ti <= len(uvs):
                        uv_out.append(uvs[ti - 1])
                indices.append(vert_map[key])

    pos_arr = np.array(pos_out, dtype=np.float32)
    norm_arr = np.array(norm_out, dtype=np.float32) if norm_out else None
    uv_arr = np.array(uv_out, dtype=np.float32) if uv_out else None
    idx_arr = np.array(indices, dtype=np.int32)

    return Mesh(pos_arr, normals=norm_arr, uvs=uv_arr, indices=idx_arr)
