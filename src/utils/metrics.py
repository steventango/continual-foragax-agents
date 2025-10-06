import polars as pl


def calculate_ewm_reward(df):
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


def calculate_mean_reward(df):
    """Calculate mean of rewards.

    Args:
        df: Polars DataFrame with 'rewards' column

    Returns:
        Polars DataFrame with 'ewm_reward' and 'mean_ewm_reward' columns added
    """
    if "rewards" not in df.columns:
        return df

    df = df.with_columns(pl.col("rewards").mean().alias("mean_reward"))
    return df


def calculate_object_traces(df):
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

        # Calculate last 1M steps occupancy
        last_1m_start = max(0, len(df) - 1000000)
        last_1m_occupancy = (
            df.slice(last_1m_start).select((pl.col("biome_id") == i).mean()).item()
        )
        df = df.with_columns(
            pl.lit(last_1m_occupancy).alias(f"biome_{i}_occupancy_last_1M")
        )

    return df
