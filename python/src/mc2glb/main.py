from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Set

import numpy as np

from . import world_io
from .nbt_decode import chunk_to_blocknames, H, Y_MIN
from .blocks import is_passable
from .atlas import PackFS, collect_textures_for_blocks
from .floodfill import flood_fill_reachable
from .mesher import extract_faces, RectFace, RectPrism
from .geometry import quad_mesh_from_rects
from .gltf_export import export_tile_glb

# optional Draco step
water_key = "__water_solid_rgba__" # must match

def bake_roi_to_tiles(
    world_zip: Path,
    pack_zip: Path,
    roi_blocks: Tuple[int, int, int, int],
    tile_chunks: int,
    out_dir: Path,
    *,
    halo_chunks: int = 1,
    make_water_transparent: bool = True,
    write_debug_atlas_png: bool = False,
    draco: bool = False,
    draco_level: int = 10,
):
    """
    The goal of this project is to take in a Minecraft world zip and a resource pack, and then output a glb for each tile in the region of interest of the world
    So generate a glb mesh and be able to later view the results in a web viewer
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[Path]] = {
        "glb": []
    }

    if draco:
        results["draco"] = []
    
    # 1 unpack and laod world
    tmp_world_root = out_dir / "_tmp_world"
    if tmp_world_root.exists():
        _rm_tree(tmp_world_root)

    world_root = world_io.unpack_zip_to_dir(world_zip, tmp_world_root)
    world = world_io.load_world(world_root)
    print(f"[debug] loading world")

    # 2 ROI -> chunk bounds + halo
    cx0, cz0, cx1, cz1 = world_io.chunk_bounds_from_block_roi(*roi_blocks)
    cx0_h, cz0_h = cx0 - halo_chunks, cz0 - halo_chunks
    cx1_h, cz1_h = cx1 + halo_chunks, cz1 + halo_chunks

    # 3 build dense global grid [H, X, Z] over ROI+halo
    X = (cx1_h - cx0_h + 1) * 16
    Z = (cz1_h - cz0_h + 1) * 16
    global_grid = np.full((H, X, Z), "minecraft:air", dtype=object)

    for (cx, cz, chunk) in world_io.iterate_chunks(world, cx0_h, cz0_h, cx1_h, cz1_h):
        grid = chunk_to_blocknames(chunk)
        ox = (cx - cx0_h) * 16
        oz = (cz - cz0_h) * 16
        global_grid[:, ox:ox+16, oz:oz+16] = grid
    print(f"[debug] finished building dense global grid")

    # 4 Seeds for BFS (spawn if in ROI) and boundary air
    seeds: List[Tuple[int, int, int]] = []
    spawn = world_io.get_spawn_xyz(world)
    if spawn is not None:
        world_x0 = cx0_h * 16
        world_z0 = cz0_h * 16
        if (world_x0 <= spawn[0] < world_x0 + X and world_z0 <= spawn[2] < world_z0 + Z):
            seeds.append(spawn)
    else:
        seeds.extend(_boundary_seeds_from_roi(global_grid, cx0_h * 16, cz0_h * 16))

    print("[debug] finished seeds")

    # 5 Floodfill reachable void
    reachable = flood_fill_reachable(global_grid, cx0_h * 16, cz0_h * 16, seeds)

    print("[debug] finished floodfill")

    # 6 Build atlas 
    used_blocks: Set[str] = set(map(str, np.unique(global_grid)))
    pack = PackFS(pack_zip)
    atlas_img, uv_map, block_tex_map = collect_textures_for_blocks(pack, used_blocks)
    if write_debug_atlas_png:
        atlas_img.save("out/debug_atlas.png")

    print("[debug] finished collect_textures_for_blocks")

    # 7 Greedy meshing across ROI+halo
    faces = extract_faces(global_grid, reachable, block_tex_map, water_key)
    opaque_rects_all: List[RectFace] = faces["opaque"]
    water_rects_all: List[RectFace] = faces["water"]

    # water_rects_all = []

    print("[debug] finished extract_faces and greedy meshing")

    # 8 Export per tile (clip rects to tile bounds). Use absolute ROI coordinates (origin_xz=(0, 0))
    for (tx0, tz0, tx1, tz1) in world_io.tiles_in_chunks(cx0, cz0, cx1, cz1, tile_chunks):
        try:
            bx0 = (tx0 - cx0_h) * 16
            bz0 = (tz0 - cz0_h) * 16
            bx1 = (tx1 - cx0_h + 1) * 16
            bz1 = (tz1 - cz0_h + 1) * 16

            rects_opaque = _clip_rects_to_tile(opaque_rects_all, bx0, bz0, bx1, bz1)
            rects_water = _clip_rects_to_tile(water_rects_all, bx0, bz0, bx1, bz1)

            # build geometry buffers (absolute ROI coords origin_xz = (0, 0))
            V, UV, N, F, _ = quad_mesh_from_rects(rects_opaque, uv_map, (0, 0), global_grid.shape)
            opaque_buffers = (V, UV, N, F)
            water_buffers = None
            if rects_water:
                VW, UW, NW, FW, _ = quad_mesh_from_rects(rects_water, uv_map, (0, 0), global_grid.shape)
                water_buffers = (VW, UW, NW, FW)
            
            out_glb = out_dir / f"tile_{tx0}_{tz0}.glb"
            export_tile_glb(out_glb, opaque_buffers, water_buffers, atlas_img, make_water_transparent=make_water_transparent)

            results["glb"].append(out_glb)

            if draco:
                raise NotImplementedError("TODO: Draco compression")
        except Exception as e:
            print(f"[debug] Exception {e} when exporting tile {tx0} {tz0}")

    
    # 9 Cleanup temp world directory
    _rm_tree(tmp_world_root)

    return results


# Internal Helpers
def _rm_tree(p: Path) -> None:
    import shutil
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass

def _boundary_seeds_from_roi(global_grid: np.ndarray, world_x0: int, world_z0: int) -> List[Tuple[int, int, int]]:
    """
    """
    H, X, Z = global_grid.shape
    seeds: List[Tuple[int, int, int]] = []

    def try_column_seed(local_x: int, local_z: int):
        for y in range(H - 2, 1, -1):
            a = global_grid[y, local_x, local_z]
            b = global_grid[y + 1, local_x, local_z]
            if is_passable(a) and is_passable(b):
                seeds.append((world_x0 + local_x, y, world_z0 + local_z))
                return

    # North/South borders
    for lx in range(X):
        try_column_seed(lx, 0)
        try_column_seed(lx, Z - 1)

    # West/East borders
    for lz in range(Z):
        try_column_seed(0, lz)
        try_column_seed(X - 1, lz)

    return seeds

def _clip_rects_to_tile(rects: Iterable[RectFace], bx0: int, bz0: int, bx1: int, bz1: int) -> List[RectFace]:
    """
    """
    out: List[RectFace] = []

    for r in rects:
        a, L, i0, j0, i1, j1, key = r

        if a == 0:
            Xp = L + 1
            if not (bx0 <= Xp < bx1): continue

            # Clip j (-z)
            jj0 = max(j0, bz0)
            jj1 = min(j1, bz1)
            if jj1 <= jj0: continue

            out.append(RectFace(a, L, i0, jj0, i1, jj1, key))

        elif a == 1:
            Xp = L
            if not (bx0 <= Xp < bx1): continue

            jj0 = max(j0, bz0)
            jj1 = min(j1, bz1)
            if jj1 <= jj0: continue

            out.append(RectFace(a, L, i0, jj0, i1, jj1, key))
        
        elif a == 2 or a == 3:
            ii0 = max(i0, bx0)
            ii1 = min(i1, bx1)
            jj0 = max(j0, bz0)
            jj1 = min(j1, bz1)
            if ii1 <= ii0 or jj1 <= jj0: continue

            out.append(RectFace(a, L, ii0, jj0, ii1, jj1, key))

        elif a == 4:
            Zp = L + 1
            if not (bz0 <= Zp < bz1): continue

            ii0 = max(i0, bx0)
            ii1 = min(i1, bx1)
            if ii1 <= ii0: continue

            out.append(RectFace(a, L, ii0, j0, ii1, j1, key))

        elif a == 5:
            Zp = L
            if not (bz0 <= Zp < bz1): continue

            ii0 = max(i0, bx0)
            ii1 = min(i1, bx1)
            if ii1 <= ii0: continue

            out.append(RectFace(a, L, ii0, j0, ii1, j1, key))

    return out

if __name__ == "__main__":
    import os
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # DEFAULT_WORLD = PROJECT_ROOT / "input" / "worlds" / "flat.zip"
    # DEFAULT_WORLD = PROJECT_ROOT / "input" / "worlds" / "first.zip"
    DEFAULT_WORLD = PROJECT_ROOT / "input" / "worlds" / "castleOnMountain.zip"
    # DEFAULT_PACK = PROJECT_ROOT / "input" / "resource_packs" / "1.21.9-Template.zip"
    DEFAULT_PACK = PROJECT_ROOT / "input" / "resource_packs" / "mad-pixels-16x-v14.zip"

    # DEFAULT_ROI = (0, 0, 16, 16) # x0, z0, x1, z1 in blocks
    # TILE_CHUNKS = 1
    DEFAULT_ROI = (-290, 176, 100, -228) # x0, z0, x1, z1 in blocks
    TILE_CHUNKS = 8
    OUT_DIR = PROJECT_ROOT / "out"

    try:
        results = bake_roi_to_tiles(
            world_zip = DEFAULT_WORLD,
            pack_zip = DEFAULT_PACK,
            roi_blocks = DEFAULT_ROI,
            tile_chunks = TILE_CHUNKS,
            out_dir = OUT_DIR,
            halo_chunks = 1,
            make_water_transparent = True,
            write_debug_atlas_png = True,
            draco = False,
            draco_level = 10,
        )
    except Exception as e:
        print(f"[mc2glb] Bake failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[mc2glb] Bake completed successfully")
    print("GLBs:")
    for p in results.get("glb", []):
        print(f" {p}")
    if "draco" in results:
        print("Draco GLBs:")
        for p in results["draco"]:
            print(f" {p}")