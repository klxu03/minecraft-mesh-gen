from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from .mesher import RectFace, RectPrism

def quad_mesh_from_rects(
    rects: List[RectFace], 
    atlas_uv: Dict[str, Tuple[float, float, float, float]], 
    origin_xz: Tuple[int, int], 
    grid_shape: Tuple[int, int, int]):
    """
    Returns packed arrays: vertices, uvs, normals, faces (indices), material_splits
    """

    ox, oz = origin_xz

    vertices: list[Tuple[float, float, float]] = []
    uvs: list[Tuple[float, float]] = []
    normals: list[Tuple[float, float, float]] = []
    faces: list[Tuple[int, int, int]] = []
    material_ranges: Dict[str, List[int]] = {}

    def add_quad(v, uv, nrm, tex_key: str):
        base = len(vertices)
        vertices.extend(v)
        uvs.extend(uv)
        normals.extend(nrm * 4)

        # two triangles per quad, CCW
        faces.append((base, base + 1, base + 2))
        faces.append((base, base + 2, base + 3))
        material_ranges.setdefault(tex_key, []).append(2)

    for r in rects:
        axis, layer, i0, j0, i1, j1, tex_key = r
        
        u0, v0, u1, v1 = atlas_uv[tex_key]
        # GLTF uses a bottom-left UV origin, but our atlas packing math
        # currently treats v=0 as the top of the image. Flip the V range so
        # that textures sampled from the atlas line up with the GLTF
        # coordinate system (otherwise the first row of tiles shows up where
        # the last row should be, and vice-versa).
        v0, v1 = 1.0 - v1, 1.0 - v0

        if axis == 0:
            x = layer + 1 + ox
            y0, y1 = i0, i1
            z0, z1 = j0, j1 # future rewrite just add oz here to z0 and z1
            vtx = [(x, y0, z0 + oz), (x, y1, z0 + oz), (x, y1, z1 + oz), (x, y0, z1 + oz)]
            uv = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
            nrm = (1.0, 0.0, 0.0)
        
        elif axis == 1:
            x = layer + ox
            y0, y1 = i0, i1
            z0, z1 = j0, j1
            vtx = [(x, y0, z1 + oz), (x, y1, z1 + oz), (x, y1, z0 + oz), (x, y0, z0 + oz)]
            uv = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
            nrm = (-1.0, 0.0, 0.0)

        elif axis == 2:
            y = layer + 1
            x0, x1 = i0, i1
            z0, z1 = j0, j1
            vtx = [(x0 + ox, y, z0 + oz), (x0 + ox, y, z1 + oz), (x1 + ox, y, z1 + oz), (x1 + ox, y, z0 + oz)]
            uv = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
            nrm = (0.0, 1.0, 0.0)
        
        elif axis == 3:
            y = layer
            x0, x1 = i0, i1
            z0, z1 = j0, j1
            vtx = [(x0 + ox, y, z1 + oz), (x0 + ox, y, z0 + oz), (x1 + ox, y, z0 + oz), (x1 + ox, y, z1 + oz)]
            uv = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
            nrm = (0.0, -1.0, 0.0)
        elif axis == 4:
            z = layer + 1
            x0, x1 = i0, i1
            y0, y1 = j0, j1
            vtx = [(x0 + ox, y0, z), (x0 + ox, y1, z), (x1 + ox, y1, z), (x1 + ox, y0, z)]
            uv = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
            nrm = (0.0, 0.0, 1.0)
        elif axis == 5:
            z = layer
            x0, x1 = i0, i1
            y0, y1 = j0, j1
            vtx = [(x1 + ox, y0, z), (x0 + ox, y0, z), (x0 + ox, y1, z), (x1 + ox, y1, z)]
            uv = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
            nrm = (0.0, 0.0, -1.0)
        
        print(f"[debug] adding quad tex_key {tex_key} with uv {uv}")
        add_quad(vtx, uv, nrm, tex_key)

    V = np.asarray(vertices, dtype=np.float32)
    UV = np.asarray(uvs, dtype=np.float32)
    N = np.asarray(normals, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)

    return V, UV, N, F, material_ranges