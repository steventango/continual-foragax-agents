import polars as pl


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
        pl.col("rewards").ewm_mean(adjust=True, alpha=1e-3).alias("ewm_reward"),
    )
    df = df.with_columns(pl.col("ewm_reward").mean().alias("mean_ewm_reward"))

    # Add ewm traces for eating objects
    df = df.with_columns(
        (pl.col("rewards") == 10)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-3)
        .alias("morel_trace_3"),
        (pl.col("rewards") == 1)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-3)
        .alias("oyster_trace_3"),
        (pl.col("rewards") == -5)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-3)
        .alias("deathcap_trace_3"),
        (pl.col("rewards") == 10)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-2)
        .alias("morel_trace_2"),
        (pl.col("rewards") == 1)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-2)
        .alias("oyster_trace_2"),
        (pl.col("rewards") == -5)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-2)
        .alias("deathcap_trace_2"),
        (pl.col("rewards") == 10)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-1)
        .alias("morel_trace_1"),
        (pl.col("rewards") == 1)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-2)
        .alias("oyster_trace_1"),
        (pl.col("rewards") == -5)
        .cast(pl.Float32)
        .ewm_mean(alpha=1e-1)
        .alias("deathcap_trace_1"),
    )

    return df


def calculate_biome_occupancy(df, window_size=30):
    """Calculate biome occupancy over time using windowed average.

    Args:
        df: Polars DataFrame with 'biome_id' column
        window_size: Window size for rolling average (default: 30)

    Returns:
        Polars DataFrame with biome occupancy columns added
    """
    if "biome_id" not in df.columns:
        return df

    # Get biome id range from data
    min_id = df.select(pl.col("biome_id").min()).item()
    max_id = df.select(pl.col("biome_id").max()).item()

    # Calculate occupancy for each biome using windowed average
    for i in range(min_id, max_id + 1):
        df = df.with_columns(
            (pl.col("biome_id") == i)
            .cast(pl.Float32)
            .rolling_mean(window_size=window_size)
            .alias(f"biome_{i}_occupancy")
        )
        df = df.with_columns(
            (pl.col("biome_id").filter(pl.col("biome_id") == i).len() / pl.len()).alias(
                f"biome_percent_{i}"
            )
        )

    return df
