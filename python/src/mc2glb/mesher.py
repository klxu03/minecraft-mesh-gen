from __future__ import annotations
import numpy as np
from typing import Dict, List, NamedTuple, Tuple
from .blocks import is_solid, is_water

class RectFace(NamedTuple):
    """
    Merged, axis-aligned rectangular face on a block grid plane

    axis:
        - 0: +X plane (normal +X), 2D coordinates are (y, z) at fixed x
        - 1: -X plane (normal -X), 2D coordinates are (y, z) at fixed x
        - 2: +Y plane (normal +Y), 2D coordinates are (x, z) at fixed y
        - 3: -Y plane (normal -Y), 2D coordinates are (x, z) at fixed y
        - 4: +Z plane (normal +Z), 2D coordinates are (x, y) at fixed z
        - 5: -Z plane (normal -Z), 2D coordinates are (x, y) at fixed z
    
    layer: integer index of the fixed coordinate (x for axes 0/1, y for 2/3, z for 4/5)
    i0, j0, i1, j1: half-open rectangle bounds on the 2D plane (rows i, cols j)
    tex_key: atlas key to use for this face
    """
    axis: int
    layer: int
    i0: int
    j0: int
    i1: int
    j1: int
    tex_key: str

def _greedy_rects(mask: np.ndarray, mat: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """
    Merge adjacent True cells (with the same material index) into maximal rectangles

    mask: (A, B) boolean plane where True indicates a face should exist
    mat: (A, B) int plane of material indices (same shape as mask)

    Returns: 
    List of rectangles as (i0, j0, i1, j1, mat_index) with half-open bounds
    """

    A, B = mask.shape
    out: List[Tuple[int, int, int, int, int]] = []
    used = np.zeros_like(mask, dtype=bool)

    for i in range(A):
        j = 0
        while j < B:
            if not mask[i, j] or used[i, j]:
                j += 1
                continue
        
            m = mat[i, j]

            # Grow width
            w = 1
            while j + w < B and mask[i, j + w] and (not used[i, j + w]) and mat[i, j + w] == m:
                w += 1

            # Grow height
            h = 1
            can_grow = True
            while i + h < A and can_grow:
                row_j = j
                while row_j < j + w:
                    if not (mask[i + h, row_j] and (not used[i + h, row_j]) and mat[i + h, row_j] == m):
                        can_grow = False
                        break
                    row_j += 1

                if can_grow:
                    h += 1

            used[i:i+h, j:j+w] = True
            out.append((i, j, i + h, j + w, int(m)))

            j += w

    return out

def extract_faces(
    global_blocks: np.ndarray,
    reachable: np.ndarray,
    tex_key_of_blocks: Dict[str, str],
    water_tex_key: str
) -> Dict[str, List[RectFace]]:
    """
    Produce merged rectangular faces for all blocks <-> reachable-air boundaries


    """
    H, X, Z = global_blocks.shape

    opaque_quads: List[RectFace] = []
    water_quads: List[RectFace] = []

    def face_tex_for(block_name: str) -> str | None:
        return tex_key_of_blocks.get(block_name, None)
    
    # Axis 0: +X
    for x in range(X - 1):
        mask = np.zeros((H, Z), dtype=bool)
        mat = np.zeros((H, Z), dtype=np.int32)
        mats: List[str] = []

        for y in range(H):
            for z in range(Z):
                a = global_blocks[y, x, z]

                if reachable[y, x + 1, z] and (is_solid(a) or is_water(a)):
                    key = water_tex_key if is_water(a) else face_tex_for(a)
                    if key is None: continue

                    mask[y, z] = True
                    if key not in mats:
                        mats.append(key)
                    mat[y, z] = mats.index(key)
        
        for (i0, j0, i1, j1, mat_idx) in _greedy_rects(mask, mat):
            key = mats[mat_idx]
            rect = RectFace(axis=0, layer=x, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

    # Axis 1: -X
    for x in range(1, X):
        mask = np.zeros((H, Z), dtype=bool)
        mat = np.zeros((H, Z), dtype=np.int32)
        mats: List[str] = []

        for y in range(H):
            for z in range(Z):
                a = global_blocks[y, x, z]
                if reachable[y, x - 1, z] and (is_solid(a) or is_water(a)):
                    key = water_tex_key if is_water(a) else face_tex_for(a)
                    if key is None: continue

                    mask[y, z] = True
                    if key not in mats:
                        mats.append(key)
                    mat[y, z] = mats.index(key)
        
        for (i0, j0, i1, j1, mat_idx) in _greedy_rects(mask, mat):
            key = mats[mat_idx]
            rect = RectFace(axis=1, layer=x - 1, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

    # Axis 2: +Y
    for y in range(H - 1):
        mask = np.zeros((X, Z), dtype=bool)
        mat = np.zeros((X, Z), dtype=np.int32)
        mats: List[str] = []

        for x in range(X):
            for z in range(Z):
                a = global_blocks[y, x, z]
                if reachable[y + 1, x, z] and (is_solid(a) or is_water(a)):
                    key = water_tex_key if is_water(a) else face_tex_for(a)
                    if key is None: continue
                    
                    mask[x, z] = True
                    if key not in mats:
                        mats.append(key)
                    mat[x, z] = mats.index(key)
        
        for (i0, j0, i1, j1, mat_idx) in _greedy_rects(mask, mat):
            key = mats[mat_idx]
            rect = RectFace(axis=2, layer=y, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

    # Axis 3: -Y
    for y in range(1, H):
        mask = np.zeros((X, Z), dtype=bool)
        mat = np.zeros((X, Z), dtype=np.int32)
        mats: List[str] = []

        for x in range(X):
            for z in range(Z):
                a = global_blocks[y, x, z]
                if reachable[y - 1, x, z] and (is_solid(a) or is_water(a)):
                    key = water_tex_key if is_water(a) else face_tex_for(a)
                    if key is None: continue
                    
                    mask[x, z] = True
                    if key not in mats:
                        mats.append(key)
                    mat[x, z] = mats.index(key)
        
        for (i0, j0, i1, j1, mat_idx) in _greedy_rects(mask, mat):
            key = mats[mat_idx]
            rect = RectFace(axis=3, layer=y - 1, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

    # Axis 4: +Z
    for z in range(Z - 1):
        mask = np.zeros((X, H), dtype=bool)
        mat = np.zeros((X, H), dtype=np.int32)
        mats: List[str] = []

        for x in range(X):
            for y in range(H):
                a = global_blocks[y, x, z]
                if reachable[y, x, z + 1] and (is_solid(a) or is_water(a)):
                    key = water_tex_key if is_water(a) else face_tex_for(a)
                    if key is None: continue
                    
                    mask[x, y] = True
                    if key not in mats:
                        mats.append(key)
                    mat[x, y] = mats.index(key)
        
        for (i0, j0, i1, j1, mat_idx) in _greedy_rects(mask, mat):
            key = mats[mat_idx]
            rect = RectFace(axis=4, layer=z, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

    # Axis 5: -Z
    for z in range(1, Z):
        mask = np.zeros((X, H), dtype=bool)
        mat = np.zeros((X, H), dtype=np.int32)
        mats: List[str] = []

        for x in range(X):
            for y in range(H):
                a = global_blocks[y, x, z]
                if reachable[y, x, z - 1] and (is_solid(a) or is_water(a)):
                    key = water_tex_key if is_water(a) else face_tex_for(a)
                    if key is None: continue
                    
                    mask[x, y] = True
                    if key not in mats:
                        mats.append(key)
                    mat[x, y] = mats.index(key)
        
        for (i0, j0, i1, j1, mat_idx) in _greedy_rects(mask, mat):
            key = mats[mat_idx]
            rect = RectFace(axis=5, layer=z - 1, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

    return {
        "opaque": opaque_quads,
        "water": water_quads
    }