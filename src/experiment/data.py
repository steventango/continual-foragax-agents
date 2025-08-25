import polars as pl


def post_process_data(df: pl.DataFrame):
    df = (
        df.with_columns(pl.col("reward").str.json_decode().cast(pl.List(pl.Float32)))
        .explode("reward")
        .with_columns(pl.int_range(0, pl.len()).over("id").alias("frame"))
        .with_columns(
            pl.col("reward")
            .ewm_mean(alpha=1e-3, adjust=False)
            .over("id")
            .alias("ewm_reward")
        )
    )

    return df
