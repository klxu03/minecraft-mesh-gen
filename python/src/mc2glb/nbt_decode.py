from __future__ import annotations
import numpy as np
import math

Y_MIN, Y_MAX = -64, 319
H = Y_MAX - Y_MIN + 1

# Toggle this to enable/disable all debug output
DEBUG = True

def unpack_palette_indices(data_longs: np.ndarray, bits: int, count: int=4096) -> np.ndarray:
    """
    Unpack 4096 paletted indices from a section's packed LongArray using bits per entry

    Post-20w17a format: Block states are aligned to 64-bit boundaries.
    Each long contains floor(64/bits) complete entries, with unused high bits.
    """
    longs = data_longs.astype(np.uint64, copy=False)
    out = np.empty(count, dtype=np.uint32)

    blocks_per_long = 64 // bits  # Integer division - how many complete blocks fit in 64 bits
    mask = (1 << bits) - 1
    i = 0

    for word in longs:
        if i >= count:
            break

        # Extract blocks_per_long values from this long
        for j in range(blocks_per_long):
            if i >= count:
                break
            out[i] = (word >> (j * bits)) & mask
            i += 1

    # Fill remaining with zeros if we ran out of data
    if i < count:
        out[i:] = 0

    return out

def _name_from_palette_entry(entry) -> str:
    # 1.18+ entries are compounds
    if isinstance(entry, dict):
        return entry.get("Name", "minecraft:air")

    # Be defensive in case a lbirary already resolved to a string
    return str(entry)

def _long_array_to_uint64(data) -> np.ndarray:
    try:
        itr = list(data)
    except TypeError:
        itr = [data]
    
    masked = [(int(v) & 0xFFFFFFFFFFFFFFFF) for v in itr]
    return np.asarray(masked, dtype=np.uint64)

def chunk_to_blocknames(chunk_nbt) -> np.ndarray:
    # Try different access methods for mcworldlib compatibility
    root = chunk_nbt
    sections = None

    # Try data_root first (raw NBT)
    if hasattr(root, 'data_root') and hasattr(root.data_root, 'get'):
        sections = root.data_root.get("sections")

    # Fall back to regular access
    if sections is None:
        sections = root.get("sections")
        if DEBUG:
            print(f"[debug] sections is none and root.get(sections) is {"none" if sections is None else "populated"}")
        if sections is None:
            level = root.get("Level", {})
            sections = level.get("sections") or level.get("Sections")

    out = np.full((H, 16, 16), "minecraft:air", dtype=object)
    if sections is None: return out

    for sec in sections:
        if not isinstance(sec, dict):
            try:
                sec = dict(sec)
            except Exception:
                continue
        
        sy = sec.get("Y")
        if sy is None:
            continue

        # DEBUG: Only process the surface section for debugging
        if int(sy) < 48/16 or int(sy) > 80/16: continue

        y0 = (int(sy) * 16) - Y_MIN
        y1 = y0 + 16
        if y1 <= 0 or y0 >= H: continue

        bs = sec.get("block_states")
        if not bs:
            continue

        pal = bs.get("palette")
        if not pal:
            continue

        palette = [_name_from_palette_entry(e) for e in pal]

        palette_excluding_air = [p for p in palette if p != 'minecraft:air']
        if DEBUG:
            print(f"[debug] pal: {pal} with len {len(pal)}")
            print(f"[debug] palette excluding air: {palette} where palette len is {len(palette)}")

        data = bs.get("data")

        # TODO possibly unnecessary if short circuit, esp since I'm excluding air maybe this was for a previous time when the whole block was air
        # Possible an entire section is just full water too. Trying to think right now maybe the short circuit is useful 
        # but if it's air, I already default everything to being air so I can just straight up skip i palette_excluding_air == 0 or something then right?
        if len(palette_excluding_air) == 0: continue
        if len(palette) == 1 or data is None:
            name = palette[0]
            ys0 = max(y0, 0)
            ys1 = min(y1, H)
            out[ys0:ys1, :, :] = name
            continue

        data_longs = _long_array_to_uint64(data)

        bits = max(4, math.ceil(math.log2(len(palette))))
        idx = unpack_palette_indices(data_longs, bits, count=16*16*16)

        cube = np.empty((16, 16, 16), dtype=object)
        pos = 0
        for yy in range(16):
            for zz in range(16):
                for xx in range(16):
                    pal_i = int(idx[pos])
                    pos += 1

                    if pal_i >= len(palette):
                        raise RuntimeError(f"Unexpected for pal_i {pal_i} to exceed len(palette) {len(palette)} for xx {xx} yy {yy} zz {zz}")

                    cube[yy, xx, zz] = palette[pal_i]
            
        ys0 = max(y0, 0)
        ys1 = min(y1, H)

        # Trim if section extends beyond our 384 window
        s0 = 0 if y0 >= 0 else -y0
        s1 = 16 if y1 <= H else 16 - (y1 - H)
        out[ys0:ys1, :, :] = cube[s0:s1, :, :]

    return out


# TODO not sure what this is for. i could probably delete this function not sure what this artifact is from 
def old_chunk_to_blocknames(chunk_nbt) -> np.ndarray:
    """
    Return a dense [H, 16, 16] array of block names for a chunk
    """
    level = None
    if "Level" in chunk_nbt:
        level = chunk_nbt["Level"]
    elif "sections" in chunk_nbt:
        level = chunk_nbt["sections"]
    else:
        print(f"[warning] chunk has no Level/sections: {chunk_nbt.keys()}")
        return np.full((H, 16, 16), "minecraft:air", dtype=object)

    # sections list contains many Y slices, each is 16x16x16
    sec_map = {}

    for sec in level.get("Sections", []):
        if "block_states" not in sec:
            continue
    
        y = int(sec["Y"])
        bs = sec["block_states"]
        palette = [entry["Name"] for entry in bs["palette"]]

        if "data" in bs:
            data = np.array(bs["data"], dtype=np.uint64)
            bits = max(4, int(np.ceil(np.log2(max(1, len(palette))))))
            idx = unpack_palette_indices(data, bits, 4096)
            names = np.array([palette[i] for i in idx], dtype=object).reshape(16, 16, 16)
        else:
            names = np.full((16, 16, 16), palette[0], dtype=object)
        
        sec_map[y] = names

    # stack sections by Y from Y_MIN to Y_MAX in 16 block sections
    layers = []
    for sy in range(Y_MIN // 16, (Y_MAX + 1) // 16):
        if sy in sec_map:
            layers.append(sec_map[sy])
        else:
            layers.append(np.full((16, 16, 16), "minecraft:air", dtype=object))

    grid = np.concatenate(layers, axis=1) # (16, 16 * #sections, 16) or (16, H, 16)
    grid = np.transpose(grid, (1, 0, 2)) # (H, 16, 16)
    return grid