
"""Factory functions for creating Foragax environment variants."""

from typing import Any, Dict, Optional, Tuple

from foragax.env import (
    Biome,
    ForagaxEnv,
    ForagaxObjectEnv,
    ForagaxRGBEnv,
    ForagaxWorldEnv,
)
from foragax.objects import (
    LARGE_MOREL,
    LARGE_OYSTER,
    MEDIUM_MOREL,
    DefaultForagaxObject,
    NormalRegenForagaxObject,
    create_weather_objects,
)

# Custom objects for modified environments
BROWN_MOREL = NormalRegenForagaxObject(
    name='brown_morel',
    reward=30.0,
    collectable=True,
    color=(63, 30, 25),  # Brown color like original morel
    mean_regen_delay=300,
    std_regen_delay=30,
)
BROWN_OYSTER = NormalRegenForagaxObject(
    name='brown_oyster',
    reward=1.0,
    collectable=True,
    color=(63, 30, 25),  # Same brown color
    mean_regen_delay=10,
    std_regen_delay=1,
)
GREEN_DEATHCAP = DefaultForagaxObject(
    name='green_deathcap',
    reward=-1.0,
    collectable=True,
    color=(0, 255, 0),  # Green
    regen_delay=(10, 10),
)
GREEN_FAKE = DefaultForagaxObject(
    name='green_fake',
    reward=0.0,
    collectable=True,
    color=(0, 255, 0),  # Green
    regen_delay=(10, 10),
)

ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ForagaxWeather-v1": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Hot biome
            Biome(start=(0, 3), stop=(15, 5), object_frequencies=(0.5, 0.0)),
            # Cold biome
            Biome(start=(0, 10), stop=(15, 12), object_frequencies=(0.0, 0.5)),
        ),
    },
    "ForagaxTwoBiomeSmall-v1": {
        "size": (16, 8),
        "aperture_size": None,
        "objects": (LARGE_MOREL, LARGE_OYSTER),
        "biomes": (
            # Morel biome
            Biome(start=(2, 2), stop=(6, 6), object_frequencies=(1.0, 0.0)),
            # Oyster biome
            Biome(start=(10, 2), stop=(14, 6), object_frequencies=(0.0, 1.0)),
        ),
    },
    "ForagaxTwoBiomeSmall-v2": {
        "size": (16, 8),
        "aperture_size": None,
        "objects": (MEDIUM_MOREL, LARGE_OYSTER),
        "biomes": (
            # Morel biome
            Biome(start=(3, 3), stop=(6, 6), object_frequencies=(1.0, 0.0)),
            # Oyster biome
            Biome(start=(11, 3), stop=(14, 6), object_frequencies=(0.0, 1.0)),
        ),
    },
    "ForagaxTwoBiome-v1": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (BROWN_MOREL, BROWN_OYSTER, GREEN_DEATHCAP, GREEN_FAKE),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.5, 0.0, 0.25, 0.0)),
            # Oyster biome
            Biome(start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.5, 0.0, 0.25)),
        ),
    },
}


def make(
    env_id: str,
    observation_type: str = "object",
    aperture_size: Optional[Tuple[int, int]] = (5, 5),
    file_index: int = 0,
) -> ForagaxEnv:
    """Create a Foragax environment.

    Args:
        env_id: The ID of the environment to create.
        observation_type: The type of observation to use. One of "object", "rgb", or "world".
        aperture_size: The size of the agent's observation aperture. If None, the default
            for the environment is used.

    Returns:
        A Foragax environment instance.
    """
    if env_id not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_id: {env_id}")

    config = ENV_CONFIGS[env_id].copy()

    config["aperture_size"] = aperture_size

    if env_id.startswith("ForagaxWeather"):
        hot, cold = create_weather_objects(file_index=file_index)
        config["objects"] = (hot, cold)

    env_class_map = {
        "object": ForagaxObjectEnv,
        "rgb": ForagaxRGBEnv,
        "world": ForagaxWorldEnv,
    }

    if observation_type not in env_class_map:
        raise ValueError(f"Unknown observation type: {observation_type}")

    env_class = env_class_map[observation_type]

    return env_class(**config)
