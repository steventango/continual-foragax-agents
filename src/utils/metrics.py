import polars as pl


def calculate_ewm_reward(df):
    """Calculate exponentially weighted moving average of rewards and traces for collected objects.

    Args:
        df: Polars DataFrame with 'rewards' column

    Returns:
        Polars DataFrame with 'ewm_reward', 'mean_ewm_reward', and object trace columns added
    """
    if "rewards" not in df.columns:
        return df

    df = df.with_columns(
        pl.col("rewards").ewm_mean(adjust=True, alpha=1e-3).alias("ewm_reward"),
    )
    df = df.with_columns(pl.col("ewm_reward").mean().alias("mean_ewm_reward"))

    # Add ewm traces for collected objects
    if "object_collected_id" in df.columns:
        min_obj = df.select(pl.col("object_collected_id").min()).item()
        max_obj = df.select(pl.col("object_collected_id").max()).item()

        for i in range(min_obj, max_obj + 1):
            if i == -1:  # Skip no collection
                continue
            df = df.with_columns(
                (pl.col("object_collected_id") == i)
                .cast(pl.Float32)
                .ewm_mean(alpha=1e-2)
                .alias(f"object_trace_{i}_2"),
                (pl.col("object_collected_id") == i)
                .cast(pl.Float32)
                .ewm_mean(alpha=1e-1)
                .alias(f"object_trace_{i}_1"),
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
