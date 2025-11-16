from __future__ import annotations
from collections import deque
from typing import Iterable, Iterator, Tuple, Any
import numpy as np

from .blocks import is_passable

def _iter_seeds(seeds: Any) -> Iterator[Tuple[int, int, int]]:
    if isinstance(seeds, (tuple, list)) and len(seeds) == 3:
        x, y, z = seeds
        yield int(x), int(y), int(z)
        return

    try:
        iterator = iter(seeds)
    except TypeError:
        return

    for s in iterator:
        if isinstance(s, (tuple, list)) and len(s) == 3:
            x, y, z = s
            yield int(x), int(y), int(z)

def flood_fill_reachable(
    global_grid: np.ndarray,
    world_x0: int,
    world_z0: int,
    seeds: Iterable[Tuple[int, int, int]] | Tuple[int, int, int] | None
) -> np.ndarray:
    """
    """
    H, X, Z = global_grid.shape
    reachable = np.zeros((H, X, Z), dtype=np.bool_)
    q = deque()

    for sx, sy, sz in _iter_seeds(seeds or []):
        lx = int(sx) - world_x0
        lz = int(sz) - world_z0
        if 0 <= lx < X and 0 <= lz < Z:
            ly = int(sy)
            if ly < 0: ly = 0
            if ly >= H: ly - H - 1
            q.append((ly, lx, lz))

    if not q:
        return reachable

    def fits(y: int, x: int, z: int) -> bool:
        if not (0 <= y < H - 1 and 0 <= x < X and 0 <= z < Z):
            return False
        
        a = global_grid[y, x, z]
        b = global_grid[y + 1, x, z]
        return is_passable(a) and is_passable(b)

    while q:
        y, x, z = q.popleft()

        if not (0 <= y < H and 0 <= x < X and 0 <= z < Z): continue

        if reachable[y, x, z]: continue
        if not fits(y, x, z): continue

        reachable[y, x, z] = True
        q.append((y + 1, x, z))
        q.append((y - 1, x, z))
        q.append((y, x + 1, z))
        q.append((y, x - 1, z))
        q.append((y, x, z + 1))
        q.append((y, x, z - 1))

    return reachable
