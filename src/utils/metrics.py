import polars as pl

from utils.constants import BIOME_DEFINITIONS


def calculate_ewm_reward(df):
    """Calculate exponentially weighted moving average of rewards and traces for eating objects.

    Args:
        df: Polars DataFrame with 'rewards' column

    Returns:
        Polars DataFrame with 'ewm_reward', 'mean_ewm_reward', 'morel_trace', 'oyster_trace', and 'deathcap_trace' columns added
    """
    if "rewards" not in df.columns:
        return df

    df = df.with_columns(
        pl.col("rewards").ewm_mean(alpha=1e-3).alias("ewm_reward"),
    )
    df = df.with_columns(pl.col("ewm_reward").mean().alias("mean_ewm_reward"))

    # Add ewm traces for eating objects
    df = df.with_columns(
        (pl.col("rewards") == 10)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-3)
        .alias("morel_trace"),
        (pl.col("rewards") == 1)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-3)
        .alias("oyster_trace"),
        (pl.col("rewards") == -5)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-3)
        .alias("deathcap_trace"),
    )

    return df


def calculate_biome_occupancy(df, window_size=30):
    """Calculate biome occupancy over time using windowed average.

    Args:
        df: Polars DataFrame with 'pos' column
        window_size: Window size for rolling average (default: 30)

    Returns:
        Polars DataFrame with biome occupancy columns added
    """
    if "pos" not in df.columns:
        return df

    # Use the global biome definitions
    biome_definitions = BIOME_DEFINITIONS.get("ForagaxTwoBiome-v1")
    if biome_definitions is None:
        return df

    biome_names = list(biome_definitions.keys()) + ["Neither"]

    x = pl.col("pos").arr.get(0)
    y = pl.col("pos").arr.get(1)

    # Chain when-then to assign biome based on first match
    biome_expr = pl.lit("Neither")
    for name, ((x1, y1), (x2, y2)) in biome_definitions.items():
        biome_expr = (
            pl.when((x >= x1) & (x < x2) & (y >= y1) & (y < y2))
            .then(pl.lit(name))
            .otherwise(biome_expr)
        )

    df = df.with_columns(biome_expr.alias("biome"))

    # Calculate occupancy for each biome using windowed average
    for name in biome_names:
        df = df.with_columns(
            (pl.col("biome") == name)
            .cast(pl.Float32)
            .rolling_mean(window_size=window_size)
            .alias(f"{name}_occupancy")
        )
        df = df.with_columns(
            (pl.col("biome").filter(pl.col("biome") == name).len() / pl.len()).alias(
                f"biome_percent_{name}"
            )
        )

    return df
