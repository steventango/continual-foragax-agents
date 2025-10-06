import tol_colors as tc

LABEL_MAP = {
    "DQN_CReLU": "DQN (CReLU)",
    "DQN_L2": "DQN (L2)",
    "DQN_L2_Init": "DQN (L2 Init)",
    "DQN_LN": "DQN (LayerNorm)",
    "DQN_Reset_Head": "DQN (Head Reset)",
    "DQN_Hare_and_Tortoise": "DQN (Hare & Tortoise)",
    "DQN_Shrink_and_Perturb": "DQN (Shrink & Perturb)",
    "DQN_privileged": "DQN (Privileged)",
    "DQN_world": "DQN (World)",
    "Search-Brown": "Search (Brown)",
    "Search-Brown-Avoid-Green": "Search (Brown, Avoid Green)",
    "Search-Morel": "Search (Morel)",
    "Search-Morel-Avoid-Green": "Search (Morel, Avoid Green)",
    "Search-Nearest": "Search (Nearest)",
    "Search-Oracle": "Search (Oracle)",
    "Search-Oyster": "Search (Oyster)",
}

# Biome definitions for different environments
BIOME_DEFINITIONS = {
    "ForagaxTwoBiomeSmall-v2": {
        "Morel": ((3, 3), (6, 6)),
        "Oyster": ((11, 3), (14, 6)),
    },
    "ForagaxTwoBiome-v1": {
        "Morel": ((3, 0), (5, 15)),
        "Oyster": ((10, 0), (12, 15)),
    },
}

# Environment mapping
ENV_MAP = {"ForagaxTwoBiomeSmall": "ForagaxTwoBiomeSmall-v2"}

# Color scheme for plotting
colorset = tc.colorsets["high_contrast"]

# Biome colors for plotting
BIOME_COLORS = {
    "Morel": colorset.blue,
    "Oyster": colorset.red,
    "Neither": colorset.yellow,
}
