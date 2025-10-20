import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl
import seaborn as sns
import tol_colors as tc

from annotate_plot import annotate_plot
from plotting_utils import (
    LABEL_MAP,
    PlottingArgumentParser,
    despine,
    filter_by_alg_aperture,
    format_metric_name,
    get_mapped_label,
    get_ylabel_mapping,
    load_data,
    parse_plotting_args,
    save_plot,
)


def main():
    parser = PlottingArgumentParser(description="Plot learning curves.")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric to plot on the y-axis (deprecated: use --metrics).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to plot on the y-axis. Multiple metrics will be plotted as subplots.",
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
    parser.add_argument(
        "--plot-all-seeds",
        action="store_true",
        help="Plot all individual seeds as separate lines instead of confidence intervals.",
    )
    parser.add_argument(
        "--subplot-by-seed",
        action="store_true",
        help="Create a grid of subplots, one per seed.",
    )

    args = parse_plotting_args(parser)

    # Handle backward compatibility: --metric takes precedence over --metrics default
    if args.metric:
        args.metrics = [args.metric]
    elif args.metrics is None:
        args.metrics = ["ewm_reward"]

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load and filter data
    df = load_data(args.experiment_path)
    logger.info(f"After load_data: {df.select('alg').unique().to_pandas()}")
    logger.info(f"DataFrame columns: {df.columns}")

    df = filter_by_alg_aperture(df, args.filter_alg_apertures)
    logger.info(
        f"After filter_by_alg_aperture: {df.select('alg').unique().to_pandas()}"
    )

    logger.info(f"Sample types present: {df.select('sample_type').unique().to_pandas()}")
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

    # # Patch temperature data for Weather environments (after filtering to reduce memory usage)
    env = df["env"][0]
    # if "Weather" in env:
    #     df = patch_temperature_data(df)
    #     logger.info("Patched temperature data into dataframe")

    logger.info(f"Final df shape: {df.shape}")
    logger.info(f"Final unique {hue_col}: {df.select(hue_col).unique().to_pandas()}")

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
        for metric in args.metrics:
            norm_df = df.filter(pl.col("alg") == args.normalize)
            norm_df = norm_df.group_by("frame").agg(pl.mean(metric).alias("norm_mean"))

            df = df.join(norm_df, on="frame", how="left")
            df = df.with_columns((pl.col(metric) / pl.col("norm_mean")).alias(metric))

    # Plotting
    num_metrics = len(args.metrics)

    # Determine subplot layout
    if args.subplot_by_seed:
        unique_seeds = df.select("seed").unique().sort("seed").to_series().to_list()
        num_seeds = len(unique_seeds)

        # Calculate grid dimensions
        import math

        ncols = math.ceil(math.sqrt(num_seeds))
        nrows = math.ceil(num_seeds / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, layout="constrained", figsize=(6 * ncols, 4 * nrows)
        )
        # Flatten axes array for easier indexing
        if num_seeds == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    elif num_metrics == 1:
        fig, axes = plt.subplots(1, 1, layout="constrained")
        axes = [axes]  # Make it a list for consistent handling
    else:
        fig, axes = plt.subplots(
            num_metrics, 1, layout="constrained", figsize=(8, 6 * num_metrics)
        )

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

    if args.subplot_by_seed:
        # Plot each seed in its own subplot
        for i, seed in enumerate(unique_seeds):
            ax = axes[i]
            seed_df = df.filter(pl.col("seed") == seed)

            for metric in args.metrics:
                sns.lineplot(
                    data=seed_df.to_pandas(),
                    x="frame",
                    y=metric,
                    hue=hue_col,
                    hue_order=hue_order,
                    palette=palette,
                    ax=ax,
                    legend="full" if i == 0 else False,
                )

            # Formatting
            if len(args.metrics) == 1:
                formatted_metric = format_metric_name(args.metrics[0])
                ylabel_map = get_ylabel_mapping(env)
                ylabel = ylabel_map.get(formatted_metric, formatted_metric)
                if args.normalize:
                    ylabel = f"Normalized {ylabel}"
            else:
                ylabel = "Value"

            ax.set_ylabel(ylabel)
            ax.set_xlabel(r"Time steps $(\times 10^6)$")
            ax.set_title(f"Seed {seed}")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
            )
            despine(ax)

            if args.ylim:
                ax.set_ylim(args.ylim)

        # Hide extra subplots if grid is not fully filled
        for j in range(num_seeds, len(axes)):
            axes[j].axis("off")

        # Handle legend
        if not args.legend:
            annotate_plot(axes[0], label_map=LABEL_MAP)
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            mapped_labels = [get_mapped_label(label, LABEL_MAP) for label in labels]
            axes[0].legend(handles, mapped_labels, title=None, frameon=False)
    else:
        # Original plotting logic for metrics
        for i, metric in enumerate(args.metrics):
            ax = axes[i]

            # Configure lineplot based on whether to show all seeds or confidence intervals
            lineplot_kwargs = {
                "data": df.to_pandas(),
                "x": "frame",
                "y": metric,
                "hue": hue_col,
                "hue_order": hue_order,
                "palette": palette,
                "ax": ax,
                "legend": "full" if i == 0 else False,
            }

            if args.plot_all_seeds:
                # Plot each seed as a separate line
                lineplot_kwargs["units"] = "seed"
                lineplot_kwargs["estimator"] = None
                lineplot_kwargs["alpha"] = 0.5  # Make individual lines semi-transparent
            else:
                # Plot mean with confidence intervals
                lineplot_kwargs["errorbar"] = ("ci", 95)

            sns.lineplot(**lineplot_kwargs)

            # Formatting
            formatted_metric = format_metric_name(metric)
            ylabel_map = get_ylabel_mapping(env)
            ylabel = ylabel_map.get(formatted_metric, formatted_metric)

            if args.normalize:
                ylabel = f"Normalized {ylabel}"
            ax.set_ylabel(ylabel)
            if i == num_metrics - 1:  # Only set x-label on the last subplot
                ax.set_xlabel(r"Time steps $(\times 10^6)$")
            else:
                ax.set_xlabel("")  # Remove x-label for non-last subplots
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
            )
            despine(ax)

            if args.ylim:
                ax.set_ylim(args.ylim)

        # Handle legend
        if not args.legend:
            annotate_plot(axes[0], label_map=LABEL_MAP)
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            mapped_labels = [get_mapped_label(label, LABEL_MAP) for label in labels]
            axes[0].legend(handles, mapped_labels, title=None, frameon=False)

    # Save plot
    if args.subplot_by_seed:
        plot_name = args.plot_name or f"{env}_by_seed"
    elif len(args.metrics) == 1:
        plot_name = args.plot_name or f"{env}_{args.metrics[0]}_curve"
    else:
        metrics_str = "_".join(args.metrics)
        plot_name = args.plot_name or f"{env}_{metrics_str}_curves"
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


if __name__ == "__main__":
    main()
