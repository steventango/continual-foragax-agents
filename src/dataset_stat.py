import logging

import polars as pl

from plotting_utils import (
    PlottingArgumentParser,
    load_data,
    parse_plotting_args,
)


def main():
    parser = PlottingArgumentParser(description="Compute dataset statistics.")

    args = parse_plotting_args(parser)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load and filter data
    df = load_data(args.experiment_path)
    logger.info(df)

    # Don't truncate the series display in printing
    pl.Config.set_tbl_formatting("UTF8_FULL")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    # Log the available algorithms and apertures
    logger.info(f"{df.select('alg', 'aperture').unique().sort(['alg', 'aperture'])}")
    # Log the available sample types
    df = df.with_columns(
        pl.col("sample_type")
        .str.extract(r"^(\d+):(\d+)(?::(\d+))?$", 1)
        .cast(pl.Int64)
        .alias("start"),
        pl.col("sample_type")
        .str.extract(r"^(\d+):(\d+)(?::(\d+))?$", 2)
        .cast(pl.Int64)
        .alias("end"),
        pl.when(pl.col("sample_type").str.contains(r"^\d+:\d+$"))
        .then(1)
        .when(pl.col("sample_type").str.contains(r"^\d+:\d+:\d+$"))
        .then(pl.col("sample_type").str.extract(r"^\d+:\d+:(\d+)", 1).cast(pl.Int64))
        .otherwise(None)
        .alias("step"),
    )
    sample_types = (
        df.select("sample_type", "start", "end", "step")
        .unique()
        .sort(["start", "end", "step"])
        .select("sample_type")
    )
    logger.info(f"{sample_types}")


if __name__ == "__main__":
    main()
