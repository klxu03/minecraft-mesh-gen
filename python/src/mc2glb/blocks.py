from __future__ import annotations

# Minimal v1 prototype classifications: treat many things as passable
PASSABLE_PREFIXES = {
    "minecraft:air",
    "minecraft:cave_air",
    "minecraft:void_air",
    "minecraft:water",
    "minecraft:seagrass",
    "minecraft:kelp",
    "minecraft:redstone_wire",
    "minecraft:vine",
    "minecraft:torch",
    "minecraft:wall_torch",
    "minecraft:flower",
    "minecraft:tall_grass"
}

def is_passable(name: str) -> bool:
    """
    Check if a block name is passable
    """
    n = name.split("[", 1)[0]
    for p in PASSABLE_PREFIXES:
        if n == p or n.startswith(p):
            return True
    
    return False

def is_solid(name: str) -> bool:
    return not is_passable(name)

def is_water(name: str) -> bool:
    return name.startswith("minecraft:water")