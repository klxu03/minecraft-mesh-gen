from __future__ import annotations
import numpy as np
from typing import Dict, List, NamedTuple, Tuple
from .blocks import is_solid, is_water

    # Toggle this to enable/disable all debug output
DEBUG = True

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

class RectPrism(NamedTuple):
    """
    A 3D rectangular prism with exposed faces

    x0,y0,z0,x1,y1,z1: bounding box of the prism (half-open)
    tex_key: material of the prism
    faces: list of RectFace objects for the exposed faces
    """
    x0: int
    y0: int
    z0: int
    x1: int
    y1: int
    z1: int
    tex_key: str
    faces: List[RectFace]

def rectfaces_to_rectprisms(rects: List[RectFace]) -> List[RectPrism]:
    """
    Group RectFace objects into RectPrism objects by identifying connected components
    of faces that touch each other in 3D space.

    This finds groups of faces that are spatially connected, which may or may not
    form complete rectangular solids (though they often will for well-formed structures).
    """
    if not rects:
        return []

    # Group faces by material first
    by_material = {}
    for rect in rects:
        if rect.tex_key not in by_material:
            by_material[rect.tex_key] = []
        by_material[rect.tex_key].append(rect)

    prisms = []

    for material, faces in by_material.items():
        if not faces:
            continue

        # Find connected components of faces that touch each other in 3D space
        components = _find_connected_face_components(faces)

        print(f"[debug] Material {material}: {len(faces)} faces -> {len(components)} connected components")

        for component_faces in components:
            if not component_faces:
                continue

            # Compute bounding box for this component
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')

            for face in component_faces:
                # Update bounds based on what blocks this face represents
                face_bounds = _get_face_block_bounds(face)
                min_x = min(min_x, face_bounds[0])
                min_y = min(min_y, face_bounds[1])
                min_z = min(min_z, face_bounds[2])
                max_x = max(max_x, face_bounds[3])
                max_y = max(max_y, face_bounds[4])
                max_z = max(max_z, face_bounds[5])

            # For now, create a prism for any connected component
            # TODO: Verify it actually forms a complete rectangular prism
            prism = RectPrism(
                x0=int(min_x), y0=int(min_y), z0=int(min_z),
                x1=int(max_x) + 1, y1=int(max_y) + 1, z1=int(max_z) + 1,  # half-open bounds
                tex_key=material,
                faces=component_faces
            )
            prisms.append(prism)

    return prisms

def _get_face_block_bounds(face: RectFace) -> Tuple[int, int, int, int, int, int]:
    """Get the 3D block bounds (min_x, min_y, min_z, max_x, max_y, max_z) that this face represents."""
    if face.axis == 0:  # +X face at layer represents blocks at x=layer
        return (face.layer, face.i0, face.j0, face.layer, face.i1 - 1, face.j1 - 1)
    elif face.axis == 1:  # -X face at layer represents blocks at x=layer
        return (face.layer, face.i0, face.j0, face.layer, face.i1 - 1, face.j1 - 1)
    elif face.axis == 2:  # +Y face at layer represents blocks at y=layer
        return (face.i0, face.layer, face.j0, face.i1 - 1, face.layer, face.j1 - 1)
    elif face.axis == 3:  # -Y face at layer represents blocks at y=layer
        return (face.i0, face.layer, face.j0, face.i1 - 1, face.layer, face.j1 - 1)
    elif face.axis == 4:  # +Z face at layer represents blocks at z=layer
        return (face.i0, face.j0, face.layer, face.i1 - 1, face.j1 - 1, face.layer)
    elif face.axis == 5:  # -Z face at layer represents blocks at z=layer
        return (face.i0, face.j0, face.layer, face.i1 - 1, face.j1 - 1, face.layer)
    else:
        return (0, 0, 0, 0, 0, 0)

def _faces_are_connected(face1: RectFace, face2: RectFace) -> bool:
    """Check if two faces touch each other in 3D space."""
    bounds1 = _get_face_block_bounds(face1)
    bounds2 = _get_face_block_bounds(face2)

    # Check if the bounding boxes overlap or touch
    overlap_x = not (bounds1[3] < bounds2[0] or bounds2[3] < bounds1[0])
    overlap_y = not (bounds1[4] < bounds2[1] or bounds2[4] < bounds1[1])
    overlap_z = not (bounds1[5] < bounds2[2] or bounds2[5] < bounds1[2])

    return overlap_x and overlap_y and overlap_z

def _find_connected_face_components(faces: List[RectFace]) -> List[List[RectFace]]:
    """Find connected components of faces using flood fill."""
    if not faces:
        return []

    # Create adjacency list
    adjacency = {i: [] for i in range(len(faces))}
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if _faces_are_connected(faces[i], faces[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Flood fill to find connected components
    visited = [False] * len(faces)
    components = []

    for i in range(len(faces)):
        if not visited[i]:
            # Start new component
            component = []
            stack = [i]

            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    component.append(faces[current])

                    # Add neighbors
                    for neighbor in adjacency[current]:
                        if not visited[neighbor]:
                            stack.append(neighbor)

            components.append(component)

    return components

def _get_expected_prism_faces(x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> List[Tuple[int, int, int, int]]:
    """Get the expected face specifications for a rectangular prism with given bounds."""
    faces = []

    # +X face at x=x1
    if x1 > x0:
        faces.append((0, x1, y0, y1, z0, z1))  # axis, layer, i0, i1, j0, j1

    # -X face at x=x0
    if x1 > x0:
        faces.append((1, x0, y0, y1, z0, z1))

    # +Y face at y=y1
    if y1 > y0:
        faces.append((2, y1, x0, x1, z0, z1))

    # -Y face at y=y0
    if y1 > y0:
        faces.append((3, y0, x0, x1, z0, z1))

    # +Z face at z=z1
    if z1 > z0:
        faces.append((4, z1, x0, x1, y0, y1))

    # -Z face at z=z0
    if z1 > z0:
        faces.append((5, z0, x0, x1, y0, y1))

    return faces

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
    print(f"[debug] Starting greedy meshing - global_blocks shape: {global_blocks.shape}, reachable shape: {reachable.shape}")
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
        
        rects = _greedy_rects(mask, mat)
        for (i0, j0, i1, j1, mat_idx) in rects:
            key = mats[mat_idx]
            rect = RectFace(axis=0, layer=x, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

        if DEBUG and len(rects) > 0:
            print(f"[debug] Axis 0 (+X) at x={x}: {len(rects)} rectangles")
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
        
        rects = _greedy_rects(mask, mat)
        for (i0, j0, i1, j1, mat_idx) in rects:
            key = mats[mat_idx]
            rect = RectFace(axis=1, layer=x, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

        if DEBUG and len(rects) > 0:
            print(f"[debug] Axis 1 (-X) at x={x}: {len(rects)} rectangles")

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
            rect = RectFace(axis=3, layer=y, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
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
            rect = RectFace(axis=5, layer=z, i0=i0, j0=j0, i1=i1, j1=j1, tex_key=key)
            (water_quads if key == water_tex_key else opaque_quads).append(rect)

    if DEBUG:
        print(f"[debug] Greedy meshing complete:")
        print(f"[debug]   Opaque rectangles: {len(opaque_quads)}")
        print(f"[debug]   Water rectangles: {len(water_quads)}")
        print(f"[debug]   Total rectangles: {len(opaque_quads) + len(water_quads)}")

        def log_rect_details(rects, rect_type):
            print(f"[debug] === {rect_type.upper()} RECT PRISMS DETAILS ===")
            print(f"[debug] Total rectangles: {len(rects)}")
            print(f"[debug] INFO: RectPrism implementation now uses proper connected component analysis.")
            print(f"[debug] Faces are grouped by material, then connected components are found.")
            print(f"[debug] Each prism represents a connected group of faces (not necessarily a complete rectangular solid).")
            print()

            # Convert RectFace list to RectPrism objects
            prisms = rectfaces_to_rectprisms(rects)

            if prisms:
                print(f"[debug] Found {len(prisms)} rectangular prisms:")

                for i, prism in enumerate(prisms):  # Show ALL prisms
                    print(f"[debug] Prism {i}: {prism.tex_key}")
                    print(f"[debug]   Bounds: ({prism.x0},{prism.y0},{prism.z0}) to ({prism.x1},{prism.y1},{prism.z1})")
                    num_blocks = (prism.x1 - prism.x0) * (prism.y1 - prism.y0) * (prism.z1 - prism.z0)
                    print(f"[debug]   Size: {(prism.x1-prism.x0)} * {(prism.y1-prism.y0)} * {(prism.z1-prism.z0)} = {num_blocks} blocks")

                    # Analyze if this is actually a 3D volume or just a 2D surface
                    x_size = prism.x1 - prism.x0
                    y_size = prism.y1 - prism.y0
                    z_size = prism.z1 - prism.z0
                    thin_dims = sum(1 for size in [x_size, y_size, z_size] if size <= 1)
                    if thin_dims >= 2:
                        print(f"[debug]   NOTE: This is a {3-thin_dims}D surface (thin in {thin_dims} dimensions)")
                    elif thin_dims == 1:
                        print(f"[debug]   NOTE: This is a wall/plate (thin in 1 dimension)")
                    else:
                        print(f"[debug]   NOTE: This is a true 3D volume")

                    print(f"[debug]   Faces: {len(prism.faces)}")

                    for face in prism.faces:  # Show ALL faces
                        axis_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
                        axis_name = axis_names[face.axis]

                        # Explain what the face coordinates mean
                        if face.axis in [0, 1]:  # X faces: i=y, j=z, layer=x
                            coord_explanation = f"Y={face.i0}..{face.i1-1}, Z={face.j0}..{face.j1-1} at X={face.layer}"
                        elif face.axis in [2, 3]:  # Y faces: i=x, j=z, layer=y
                            coord_explanation = f"X={face.i0}..{face.i1-1}, Z={face.j0}..{face.j1-1} at Y={face.layer}"
                        elif face.axis in [4, 5]:  # Z faces: i=x, j=y, layer=z
                            coord_explanation = f"X={face.i0}..{face.i1-1}, Y={face.j0}..{face.j1-1} at Z={face.layer}"
                        else:
                            coord_explanation = "unknown"

                        face_blocks = (face.i1 - face.i0) * (face.j1 - face.j0)
                        print(f"[debug]     {axis_name} face at layer {face.layer}: bounds ({face.i0},{face.j0})->({face.i1},{face.j1})")
                        print(f"[debug]         Meaning: {coord_explanation}")
                        print(f"[debug]         Covers: {(face.i1 - face.i0)} × {(face.j1 - face.j0)} = {face_blocks} block faces (merged by greedy meshing)")
                    print()
            else:
                print(f"[debug] No rectangular prisms found!")

        if len(opaque_quads) > 0:
            log_rect_details(opaque_quads, "opaque")
        if len(water_quads) > 0:
            log_rect_details(water_quads, "water")

    return {
        "opaque": opaque_quads,
        "water": water_quads
    }