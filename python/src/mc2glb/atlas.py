# Minimal resource-pack loader and texture atlas builder

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
from PIL import Image
import zipfile
import math

DEBUG = False

class PackFS:
    """
    Pack File System: Thin wrapper over a resource pack ZIP
    """

    def __init__(self, pack_zip: Path):
        self.z = zipfile.ZipFile(pack_zip)
        # self.list_all()

    def open_image(self, rel: str) -> Image.Image | None:
        """
        rel like assets/minecraft/textures/block/stone.png
        """
        try:
            with self.z.open(rel) as f:
                return Image.open(f).convert("RGBA")
        except KeyError:
            return None

    def find_block_png(self, block_id: str) -> str | None:
        name = block_id.split(":", 1)[1]

        cand = f"assets/minecraft/textures/block/{name}.png"
        if self.exists(cand): return cand
        print(f"[debug] cand {cand} does not exist")

        cand_top = f"assets/minecraft/textures/block/{name}_top.png"
        # cand_top = f"assets/minecraft/textures/block/azalea_top.png"
        print(f"[debug] attempting to use cand_top {cand_top}")
        if self.exists(cand_top): return cand_top

        print(f"[debug] cand {cand} and cand_top does not exist, using fallback")
        fallback = "assets/minecraft/textures/block/diamond_block.png"
        return fallback if self.exists(fallback) else None

    def exists(self, rel:str) -> bool:
        try:
            self.z.getinfo(rel)
            return True
        except KeyError:
            return False

    def list_all(self):
        for file in self.z.filelist:
            print(f"File {file}")

def build_atlas(images: Dict[str, Image.Image], tile_px=16) -> Tuple[Image.Image, Dict[str, Tuple[float, float, float, float]]]:
    """
    Pack images of identical size (16 x 16 in vanilla) onto a square grid atlas
    Returns (atlas_image, {key: (u0, v0, u1, v1)})
    """

    keys = sorted(images.keys())
    n = len(keys)
    grid = int(math.ceil(math.sqrt(max(1, n))))
    atlas = Image.new("RGBA", (grid * tile_px, grid * tile_px), (0, 0, 0, 0))
    uv = {}
    i = 0

    for key in keys:
        img = images[key]
        x = (i % grid) * tile_px
        y = (i // grid) * tile_px
        atlas.paste(img, (x, y))

        u0 = x / atlas.width
        v0 = y / atlas.height
        u1 = (x + tile_px) / atlas.width
        v1 = (y + tile_px) / atlas.height
        uv[key] = (u0, v0, u1, v1)
        i += 1

    return atlas, uv

def collect_textures_for_blocks(pack: PackFS, block_ids: set[str], tile_px = 16) -> Tuple[Image.Image, Dict[str, Tuple[float, float, float, float]], Dict[str, str]]:
    """
    For prototype v1, map every block to a single texture
    Returns atlas image, UV map, and mapping block_id -> atlas_key
    
    In the future we will need to improve with a cube_all, getting cube_bottom_top, orientable etc. instead of same texture for all faces
    """

    tex_map: Dict[str, str] = {}
    image_map: Dict[str, Image.Image] = {}

    for bid in block_ids:
        if DEBUG:
            print(f"[debug] bid is {bid}")
        rel = pack.find_block_png(bid)
        if not rel: continue

        if rel not in image_map:
            img = pack.open_image(rel)
            if img is None: continue

            image_map[rel] = img

        tex_map[bid] = rel

    # Maybe I can just use the texture pack's water block
    water_key = "__water_solid_rgba__"
    water_img = Image.new("RGBA", (16, 16), (63, 118, 228, 200))
    image_map[water_key] = water_img

    atlas, uv = build_atlas(image_map, tile_px)
    return atlas, uv, tex_map