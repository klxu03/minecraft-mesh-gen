from __future__ import annotations
from pathlib import Path
import zipfile
from typing import Tuple, Iterator
import mcworldlib as mc

OVERWORLD = mc.OVERWORLD

def unpack_zip_to_dir(zip_path: Path, dest_dir: Path) -> Path:
    """
    Unzips a world.zip to dest_dir and returns the dir containing level.dat
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    for p in dest_dir.rglob("level.dat"):
        return p.parent
    raise FileNotFoundError(f"level.dat not found in {dest_dir} after unzip")

def load_world(unpacked_world_dir: Path) -> mc.World:
    """
    Loads a world from an unpacked world directory
    """
    return mc.load(str(unpacked_world_dir))

def get_spawn_xyz(world) -> Tuple[int, int, int] | None:
    """
    Gets the spawn xyz from the world level.dat
    """
    data = world.level["Data"]
    x = int(data.get("SpawnX", 0))
    y = int(data.get("SpawnY", 128))
    z = int(data.get("SpawnZ", 0))
    return (x, y, z)

def chunk_bounds_from_block_roi(x0: int, z0: int, x1: int, z1: int) -> Tuple[int, int, int, int]:
    """
    Converts block-space ROI into inclusive chunk coordinates
    """
    # swap if x0 > x1 or z0 > z1
    if x0 > x1: x0, x1 = x1, x0
    if z0 > z1: z0, z1 = z1, z0

    cx0 = x0 // 16
    cx1 = (x1 - 1) // 16
    cz0 = z0 // 16
    cz1 = (z1 - 1) // 16
    return (cx0, cz0, cx1, cz1)

def tiles_in_chunks(cx0: int, cz0: int, cx1: int, cz1: int, tile_chunks:int=8):
    """
    Iterates over tiles in chunks
    """
    for tz in range(cz0, cz1 + 1, tile_chunks):
        for tx in range(cx0, cx1 + 1, tile_chunks):
            yield (tx, tz, min(cx1, tx + tile_chunks - 1), min(cz1, tz + tile_chunks - 1))

def iterate_chunks(world, cx0:int, cz0:int, cx1:int, cz1:int) -> Iterator[Tuple[int, int, object]]:
    regions = world.regions[OVERWORLD]

    for cz in range(cz0, cz1 + 1):
        for cx in range(cx0, cx1 + 1):
            chunk = world.get_chunk((cx, cz))
            yield (cx, cz, chunk)