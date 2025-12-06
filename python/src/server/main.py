from __future__ import annotations

import os, re, time
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Dict, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

TILE_REGEX = re.compile(r"tile_(-?\d+)_(-?\d+)\.glb", re.IGNORECASE)
TILES_DIR = (Path(__file__).resolve().parent / ".." / ".." / "out").resolve()
tiles: Dict[Tuple[int, int], str] = {}

def scan_tiles():
    tiles.clear()
    if not os.path.isdir(TILES_DIR):
        raise FileNotFoundError(f"Tiles directory {TILES_DIR} not found")

    for file in os.listdir(TILES_DIR):
        print(f"file: {file}")
        m = TILE_REGEX.fullmatch(file)
        if not m:
            continue

        cx, cz = int(m.group(1)), int(m.group(2))
        print(f"cx: {cx}, cz: {cz}")
        tiles[(cx, cz)] = str(Path(TILES_DIR) / file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    scan_tiles()
    yield

app = FastAPI(title="Minecraft Mesh Generator Server", version="v1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "tiles": len(tiles), "time": int(time.time())}

@app.get("/tile/{cx}/{cz}")
def get_tile(cx: int, cz: int):
    key = (cx, cz)
    print(f"key: {key}")
    path = tiles.get(key)
    if path is None or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Tile not found: {cx},{cz}")

    return FileResponse(path, filename=os.path.basename(path), media_type="model/gltf-binary")

if __name__ == "__main__":
    # Use an import string so reload works correctly.
    uvicorn.run("server.main:app", host="127.0.0.1", port=8000, reload=True)