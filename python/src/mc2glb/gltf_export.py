from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy

import trimesh
from PIL import Image
from pygltflib import GLTF2, Material, PbrMetallicRoughness

from .mesher import RectFace, RectPrism
from .geometry import quad_mesh_from_rects
import numpy as np

def _build_trimesh(V, UV, N, F, atlas_image: Image.Image) -> trimesh.Trimesh:
    img_rgba = atlas_image.convert("RGBA")
    material = trimesh.visual.material.SimpleMaterial(image=img_rgba)
    vis = trimesh.visual.texture.TextureVisuals(uv=UV, material=material)

    mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_normals=N, visual=vis, process=False)
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'to_pbr'):
        mesh.visual.material = mesh.visual.material.to_pbr()
    return mesh

def export_tile_glb(out_path: Path,
    opaque_buffers: Tuple, # (V, UV, N, F)
    water_buffers: Optional[Tuple], # (V, UV, N, F) or None
    atlas_image: Image.Image,
    make_water_transparent: bool = True):
    """
    """
    V, UV, N, F = opaque_buffers
    mesh_opaque = _build_trimesh(V, UV, N, F, atlas_image)

    scene = trimesh.Scene()
    scene.add_geometry(mesh_opaque, node_name="opaque")

    if water_buffers is not None:
        VW, UW, NW, FW = water_buffers
        mesh_water = _build_trimesh(VW, UW, NW, FW, atlas_image)
        scene.add_geometry(mesh_water, node_name="water")

    # Write raw GLB
    glb_bytes = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    out_path.write_bytes(glb_bytes)

    # make water transparent by editing materials used by node water
    if make_water_transparent and water_buffers is not None:
        _patch_water_material_alpha(out_path, target_node_name="water", alpha=0.78)

    _ensure_all_materials_double_sided(out_path)


def export_tile_glb_from_rects(
    out_path: Path,
    rects_opaque: List[RectFace],
    rects_water: Optional[List[RectFace]],
    atlas_image: Image.Image,
    atlas_uv: Dict[str, Tuple[float, float, float, float]],
    origin_xz: Tuple[int, int],
    make_water_transparent: bool = True
):
    """
    """
    V, UV, N, F, _ = quad_mesh_from_rects(rects_opaque, atlas_uv, origin_xz)
    opaque = (V, UV, N, F)
    water = None

    if rects_water:
        VW, UW, NW, FW, _ = quad_mesh_from_rects(rects_water, atlas_uv, origin_xz)
        water = (VW, UW, NW, FW)
    export_tile_glb(out_path, opaque, water, atlas_image, make_water_transparent)

def _patch_water_material_alpha(glb_path: Path, target_node_name: str = "water", alpha: float = 0.78):
    """
    """
    gltf = GLTF2().load_binary(glb_path.as_posix())

    if gltf.nodes is None:
        return

    meshes = gltf.meshes or []
    materials = gltf.materials or []
    materials_changed = False

    for node_index, node in enumerate(gltf.nodes):
        if node.name != target_node_name:
            continue

        if node.mesh is None:
            continue

        mesh = meshes[node.mesh]

        for prim in mesh.primitives:
            mi = prim.material
            if mi is None:
                # create a new transparent PBR material
                m = Material(
                    name=f"{target_node_name}_mat",
                    pbrMetallicRoughness=PbrMetallicRoughness(
                        baseColorFactor=[1.0, 1.0, 1.0, alpha],
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                    alphaMode="BLEND",
                    doubleSided=True,
                )
                if gltf.materials is None:
                    gltf.materials = []
                gltf.materials.append(m)
                prim.material = len(gltf.materials) - 1
                materials_changed = True
            else:
                # clonse existing material and make it blend with alpha
                base = materials[mi]
                cloned = copy.deepcopy(base)
                if cloned.pbrMetallicRoughness is None:
                    cloned.pbrMetallicRoughness = PbrMetallicRoughness()
                
                if cloned.pbrMetallicRoughness.baseColorFactor is None:
                    cloned.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, alpha]
                else:
                    bcf = cloned.pbrMetallicRoughness.baseColorFactor
                    bcf = [bcf[0], bcf[1], bcf[2], alpha]
                    cloned.pbrMetallicRoughness.baseColorFactor = bcf
                
                cloned.alphaMode = "BLEND"
                cloned.doubleSided = True

                gltf.materials.append(cloned)
                prim.material = len(gltf.materials) - 1
                materials_changed = True

    if materials_changed:
        gltf.save_binary(glb_path.as_posix())


def _ensure_all_materials_double_sided(glb_path: Path) -> None:
    """
    Ensure every material in the GLB is marked double-sided so the mesh is visible from inside structures.
    """
    gltf = GLTF2().load_binary(glb_path.as_posix())

    materials = gltf.materials or []
    changed = False
    for mat in materials:
        if not mat.doubleSided:
            mat.doubleSided = True
            changed = True

    if changed:
        gltf.save_binary(glb_path.as_posix())