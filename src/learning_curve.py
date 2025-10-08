import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl
import seaborn as sns
import tol_colors as tc

from annotate_plot import annotate_plot
from plotting_utils import (
    LABEL_MAP,
    YLABEL_MAP,
    PlottingArgumentParser,
    despine,
    filter_by_alg_aperture,
    format_metric_name,
    get_mapped_label,
    load_data,
    parse_plotting_args,
    save_plot,
)


def main():
    parser = PlottingArgumentParser(description="Plot learning curves.")
    parser.add_argument(
        "--metric",
        type=str,
        default="ewm_reward",
        help="Metric to plot on the y-axis.",
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="every",
        help="Sample type to filter from the data.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Y-axis limits for the plot.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1000,
        help="Minimum frame to include in the plot (default: 1000 to skip warm-up)",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Use legend instead of auto-labeling",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Algorithm to normalize against.",
    )

    args = parse_plotting_args(parser)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load and filter data
    df = load_data(args.experiment_path)
    logger.info(f"After load_data: {df.select('alg').unique().to_pandas()}")

    df = filter_by_alg_aperture(df, args.filter_alg_apertures)
    logger.info(
        f"After filter_by_alg_aperture: {df.select('alg').unique().to_pandas()}"
    )

    df = df.filter(pl.col("sample_type") == args.sample_type)
    logger.info(f"After sample_type filter: {df.select('alg').unique().to_pandas()}")

    if args.filter_algs:
        df = df.filter(pl.col("alg").is_in(args.filter_algs))

    hue_col = "alg"
    if args.filter_alg_apertures:
        df = df.with_columns(
            pl.when(pl.col("aperture").is_not_null())
            .then(pl.col("alg") + ":" + pl.col("aperture").cast(pl.Utf8))
            .otherwise(pl.col("alg"))
            .alias("alg_ap")
        )
        hue_col = "alg_ap"
        logger.info(
            f"After creating alg_ap: {df.select('alg_ap').unique().to_pandas()}"
        )

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    df = df.filter(pl.col("frame") >= args.start_frame)

    logger.info(f"Final df shape: {df.shape}")
    logger.info(f"Final unique {hue_col}: {df.select(hue_col).unique().to_pandas()}")

    env = df["env"][0]

    # Sort by order specified in arguments
    if args.filter_alg_apertures:
        alg_ap_order = {ap: i for i, ap in enumerate(args.filter_alg_apertures)}
        df = df.with_columns(pl.col("alg_ap").replace(alg_ap_order).alias("order_col"))
        df = df.sort("order_col")
    elif args.filter_algs:
        alg_order = {alg: i for i, alg in enumerate(args.filter_algs)}
        df = df.with_columns(pl.col("alg").replace(alg_order).alias("order_col"))
        df = df.sort("order_col")

    # Normalization
    if args.normalize:
        norm_df = df.filter(pl.col("alg") == args.normalize)
        norm_df = norm_df.group_by("frame").agg(pl.mean(args.metric).alias("norm_mean"))

        df = df.join(norm_df, on="frame", how="left")
        df = df.with_columns(
            (pl.col(args.metric) / pl.col("norm_mean")).alias(args.metric)
        )

    # Plotting
    fig, ax = plt.subplots(layout="constrained")

    hue_order = df.select(hue_col).unique().to_series().to_list()

    # Create color palette matching the order in filter_alg_apertures
    vibrant_colors = list(tc.colorsets["vibrant"])
    if args.filter_alg_apertures:
        # Map colors to alg-aperture combinations in the order specified
        palette = {
            alg_ap: vibrant_colors[i % len(vibrant_colors)]
            for i, alg_ap in enumerate(args.filter_alg_apertures)
        }
    elif args.filter_algs:
        # Map colors to algorithms in the order specified
        palette = {
            alg: vibrant_colors[i % len(vibrant_colors)]
            for i, alg in enumerate(args.filter_algs)
        }
    else:
        # Use default palette ordering
        palette = None

    sns.lineplot(
        data=df.to_pandas(),
        x="frame",
        y=args.metric,
        hue=hue_col,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        errorbar=("ci", 95),
        legend="full",
    )

    # Formatting
    formatted_metric = format_metric_name(args.metric)
    ylabel = YLABEL_MAP.get(formatted_metric, formatted_metric)
    if args.normalize:
        ylabel = f"Normalized {ylabel}"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"Time steps $(\times 10^6)$")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x // 1000000)}")
    )
    despine(ax)

    if args.ylim:
        ax.set_ylim(args.ylim)

    if not args.legend:
        annotate_plot(ax, label_map=LABEL_MAP)
    else:
        handles, labels = ax.get_legend_handles_labels()
        mapped_labels = [get_mapped_label(label, LABEL_MAP) for label in labels]
        plt.legend(handles, mapped_labels, title=None, frameon=False)

    # Save plot
    plot_name = args.plot_name or f"{env}_{args.metric}_curve"
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


if __name__ == "__main__":
    main()
