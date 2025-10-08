import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from plotting_utils import (
    LABEL_MAP,
    TWO_BIOME_COLORS,
    WEATHER_BIOME_COLORS,
    PlottingArgumentParser,
    despine,
    filter_by_alg_aperture,
    get_biome_mapping,
    get_mapped_label,
    load_data,
    parse_plotting_args,
    save_plot,
)

# Mappings
SAMPLE_TYPE_MAP = {
    "999000:1000000:500": "Early learning",
    "4999000:5000000:500": "Mid learning",
    "9999000:10000000:500": "Late learning",
}


def main():
    parser = PlottingArgumentParser(description="Plot biome occupancy as stacked bars.")
    parser.add_argument("--sample-types", nargs="*", help="Sample types to plot.")
    parser.add_argument(
        "--window", type=int, default=1000, help="Occupancy window size."
    )
    parser.add_argument(
        "--sort-seeds",
        action="store_true",
        help="Sort seeds by a metric before plotting.",
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, help="Set x-axis limits for the horizontal bars."
    )

    args = parse_plotting_args(parser)

    # Load and filter data
    df = load_data(args.experiment_path)
    df = filter_by_alg_aperture(df, args.filter_alg_apertures)

    if args.sample_types:
        df = df.filter(pl.col("sample_type").is_in(args.sample_types))

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    env = df["env"][0]

    # Determine biome mappings
    biome_mapping = get_biome_mapping(env)
    if "TwoBiome" in env:
        biome_colors = TWO_BIOME_COLORS
        biome_order = ["Morel", "Neither", "Oyster"]
    elif "Weather" in env:
        biome_colors = WEATHER_BIOME_COLORS
        biome_order = ["Hot", "Neither", "Cold"]
    else:
        raise ValueError(f"Unknown biome mapping for environment: {env}")

    available_biomes = sorted(df["biome_id"].unique())
    biome_metrics = [f"biome_{b}_occupancy_{args.window}" for b in available_biomes]
    biome_names = [biome_mapping[b] for b in available_biomes]

    # Reorder based on predefined order
    ordered_indices = [
        biome_names.index(name) for name in biome_order if name in biome_names
    ]
    biome_metrics = [biome_metrics[i] for i in ordered_indices]
    biome_names = [biome_names[i] for i in ordered_indices]

    main_alg_apertures = sorted(df.select(["alg", "aperture"]).unique().rows())
    sample_types_list = args.sample_types or df["sample_type"].unique().to_list()

    # Create figure
    nrows = len(main_alg_apertures)
    ncols = len(sample_types_list)
    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey=False, layout="constrained", squeeze=False
    )

    # Aggregate data
    agg_data = df.group_by(["alg", "aperture", "sample_type", "seed"]).agg(
        [pl.mean(metric).alias(metric) for metric in biome_metrics]
    )

    # Plotting
    for i, (alg, aperture) in enumerate(main_alg_apertures):
        for j, sample_type in enumerate(sample_types_list):
            ax = axs[i, j]

            if aperture is None:
                plot_df = agg_data.filter(
                    (pl.col("alg") == alg)
                    & pl.col("aperture").is_null()
                    & (pl.col("sample_type") == sample_type)
                )
            else:
                plot_df = agg_data.filter(
                    (pl.col("alg") == alg)
                    & (pl.col("aperture") == aperture)
                    & (pl.col("sample_type") == sample_type)
                )

            if plot_df.is_empty():
                continue

            # Sort seeds if requested
            if args.sort_seeds:
                # Convert to dictionary format for sorting
                seed_data = {}
                for row in plot_df.iter_rows(named=True):
                    seed = row["seed"]
                    seed_data[seed] = {metric: row[metric] for metric in biome_metrics}

                # Sort seeds based on biome occupancy
                if "TwoBiome" in env:
                    morel_metric = f"biome_0_occupancy_{args.window}"
                    oyster_metric = f"biome_1_occupancy_{args.window}"
                    sorted_seeds = sorted(
                        seed_data.keys(),
                        key=lambda s: (
                            -seed_data[s].get(morel_metric, 0.0),  # descending morel
                            seed_data[s].get(oyster_metric, 0.0),  # ascending oyster
                        ),
                    )
                elif "Weather" in env:
                    hot_metric = f"biome_0_occupancy_{args.window}"
                    cold_metric = f"biome_1_occupancy_{args.window}"
                    sorted_seeds = sorted(
                        seed_data.keys(),
                        key=lambda s: (
                            -seed_data[s].get(hot_metric, 0.0),  # descending hot
                            seed_data[s].get(cold_metric, 0.0),  # ascending cold
                        ),
                    )
                else:
                    sorted_seeds = plot_df["seed"].to_list()

                # Reorder plot_df based on sorted seeds
                seed_order = {seed: idx for idx, seed in enumerate(sorted_seeds)}
                plot_df = plot_df.with_columns(
                    pl.col("seed").replace(seed_order).alias("seed_order")
                )
                plot_df = plot_df.sort("seed_order")

            bottom = np.zeros(len(plot_df))

            # Create y-positions (0, 1, 2, ...) for each seed
            y_positions = np.arange(len(plot_df))

            for metric, name in zip(biome_metrics, biome_names, strict=True):
                values = plot_df[metric].to_numpy()
                color = biome_colors[name]
                ax.barh(
                    y_positions,
                    values,
                    left=bottom,
                    label=name,
                    color=color,
                    height=1,
                    edgecolor=color,
                )
                bottom += values

            # Set y-axis limits and invert so first seed is on top
            ax.set_ylim(-0.5, len(plot_df) - 0.5)
            ax.invert_yaxis()

            despine(ax)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            ax.tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )
            ax.grid(False)

    # Formatting
    for i, (alg, aperture) in enumerate(main_alg_apertures):
        if aperture is not None:
            temp_label = f"{alg}:{aperture}"
        else:
            temp_label = alg
        label = get_mapped_label(temp_label, LABEL_MAP)
        label = label.replace(" (", "\n(")

        if label:
            axs[i, 0].set_ylabel(label, rotation=0, ha="right", va="center")

    for j, sample_type in enumerate(sample_types_list):
        axs[-1, j].set_xlabel(SAMPLE_TYPE_MAP.get(sample_type, sample_type))

    if args.ylim:
        plt.setp(axs, xlim=args.ylim)
    else:
        plt.setp(axs, xlim=(0, 1))

    # Legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=biome_colors[n], label=n)
        for n in biome_names
    ]
    fig.legend(
        handles=legend_elements,
        loc="outside upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        ncol=len(biome_names),
    )

    plot_name = args.plot_name or f"{env}_biome_stacked_bar"
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


if __name__ == "__main__":
    main()
