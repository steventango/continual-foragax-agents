import polars as pl


def calculate_ewm_reward(df: pl.DataFrame):
    """Calculate exponentially weighted moving average of rewards.

    Args:
        df: Polars DataFrame with 'rewards' column

    Returns:
        Polars DataFrame with 'ewm_reward' and 'mean_ewm_reward' columns added
    """
    if "rewards" not in df.columns:
        return df

    df = df.with_columns(
        pl.col("rewards").ewm_mean(alpha=1e-3, adjust=True).alias("ewm_reward"),
    )
    df = df.with_columns(pl.col("ewm_reward").mean().alias("mean_ewm_reward"))

    return df


def calculate_mean_reward(df: pl.DataFrame):
    """Calculate mean of rewards.

    Args:
        df: Polars DataFrame with 'rewards' column

    Returns:
        Polars DataFrame with 'rolling_reward' and 'mean_reward' columns added
    """
    if "rewards" not in df.columns:
        return df

    window_sizes = [10, 100, 1000, 10000, 100000, 1000000]

    for window_size in window_sizes:
        df = df.with_columns(
            pl.col("rewards")
            .rolling_mean(window_size=window_size)
            .alias(f"rolling_reward_{window_size}"),
        )

    df = df.with_columns(pl.col("rewards").mean().alias("mean_reward"))
    return df


def calculate_object_traces(df: pl.DataFrame):
    """Calculate exponentially weighted moving traces for collected objects.

    Args:
        df: Polars DataFrame with 'object_collected_id' column

    Returns:
        Polars DataFrame with object trace columns added
    """
    if "object_collected_id" not in df.columns:
        return df

    max_obj = df.select(pl.col("object_collected_id").max()).item()

    for i in range(max_obj + 1):
        df = df.with_columns(
            (pl.col("object_collected_id") == i)
            .cast(pl.Float32)
            .ewm_mean(alpha=1e-2, adjust=True)
            .alias(f"object_trace_{i}_2"),
            (pl.col("object_collected_id") == i)
            .cast(pl.Float32)
            .ewm_mean(alpha=1e-1, adjust=True)
            .alias(f"object_trace_{i}_1"),
        )

    return df


def calculate_biome_occupancy(df: pl.DataFrame):
    """Calculate biome occupancy over time using windowed average.

    Args:
        df: Polars DataFrame with 'biome_id' column

    Returns:
        Polars DataFrame with biome occupancy columns added
    """
    if "biome_id" not in df.columns:
        return df

    window_sizes = [1, 10, 100, 1000, 10000, 100000, 500000, 1000000]

    # Get biome id range from data
    min_id = df.select(pl.col("biome_id").min()).item()
    max_id = df.select(pl.col("biome_id").max()).item()

    # Calculate occupancy for each biome using windowed average
    for i in range(min_id, max_id + 1):
        df = df.with_columns(
            (pl.col("biome_id").filter(pl.col("biome_id") == i).len() / pl.len()).alias(
                f"biome_percent_{i}"
            )
        )
        for window_size in window_sizes:
            df = df.with_columns(
                (pl.col("biome_id") == i)
                .cast(pl.Float32)
                .rolling_mean(window_size=window_size)
                .alias(f"biome_{i}_occupancy_{window_size}")
            )

    return df
