import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import bootstrap

from plotting_utils import (
    PlottingArgumentParser,
    despine,
    format_metric_name,
    load_data,
    parse_plotting_args,
    save_plot,
)


def parse_bars(
    bar_strings: list[str] | None,
) -> list[tuple[str, int | None, str, list[int] | None]] | None:
    if not bar_strings:
        return None

    bars = []
    for bar_str in bar_strings:
        parts = bar_str.split("|")
        alg = parts[0]
        aperture = int(parts[1]) if parts[1] else None
        sample_type = parts[2]
        seeds = [int(s) for s in parts[3].split(",")] if parts[3] else None
        bars.append((alg, aperture, sample_type, seeds))
    return bars


def main():
    parser = PlottingArgumentParser(description="Plot mean reward bar plots.")
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_reward",
        help="Metric to plot.",
    )
    parser.add_argument(
        "--sort-by-metric",
        action="store_true",
        help="Sort bars by metric value.",
    )
    parser.add_argument(
        "--bars",
        nargs="*",
        help="Bar specifications in format 'alg|aperture|sample_type|seeds'",
    )
    args = parse_plotting_args(parser)
    bars = parse_bars(args.bars)

    df = load_data(args.experiment_path)
    env = df["env"][0]

    if not bars:
        unique_algs = df["alg"].unique().to_list()
        bars = [(alg, None, "every", None) for alg in unique_algs]

    plot_data = []
    for alg, aperture, sample_type, seeds in bars:
        bar_df = df.filter(pl.col("sample_type") == sample_type)
        if aperture is not None:
            bar_df = bar_df.filter(pl.col("aperture") == aperture)
        if seeds is not None:
            bar_df = bar_df.filter(pl.col("seed").is_in(seeds))

        bar_df = bar_df.filter(pl.col("alg") == alg)

        # Get data at the last frame
        last_frame = bar_df["frame"].max()
        bar_df = bar_df.filter(pl.col("frame") == last_frame)

        label = alg
        if aperture is not None:
            label += f" (A: {aperture})"

        metric_values = bar_df[args.metric].to_numpy()
        mean_metric = np.mean(metric_values)

        # Bootstrap for confidence intervals
        res = bootstrap((metric_values,), np.mean, confidence_level=0.95)
        ci_low, ci_high = res.confidence_interval

        plot_data.append(
            {
                "label": label,
                "metric": mean_metric,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    plot_df = pl.DataFrame(plot_data)

    if args.sort_by_metric:
        plot_df = plot_df.sort("metric", descending=True)

    # Plotting
    fig, ax = plt.subplots(layout="constrained")

    yerr = [
        plot_df["metric"] - plot_df["ci_low"],
        plot_df["ci_high"] - plot_df["metric"],
    ]

    sns.barplot(
        x="label",
        y="metric",
        data=plot_df.to_pandas(),
        ax=ax,
        palette="vibrant",
    )
    ax.errorbar(
        x=plot_df["label"],
        y=plot_df["metric"],
        yerr=yerr,
        fmt="none",
        c="black",
        capsize=3,
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel(format_metric_name(args.metric))
    ax.set_title(f"{env} - {format_metric_name(args.metric)}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    despine(ax)

    plot_name = args.plot_name or f"{env}_{args.metric}_bar"
    save_plot(
        fig,
        args.experiment_path,
        plot_name,
        args.save_type,
        width=len(bars) / 2,
        height_ratio=1 / 3 / len(bars),
    )


if __name__ == "__main__":
    main()
