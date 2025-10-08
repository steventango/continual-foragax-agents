import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from plotting_utils import (
    LABEL_MAP,
    TWO_BIOME_COLORS,
    WEATHER_BIOME_COLORS,
    PlottingArgumentParser,
    despine,
    filter_by_alg_aperture,
    get_biome_mapping,
    load_data,
    parse_plotting_args,
    save_plot,
)


def main():
    parser = PlottingArgumentParser(description="Plot grouped biome occupancy bars.")
    parser.add_argument("--sample-type", type=str, default="end", help="Sample type to plot.")
    parser.add_argument("--window", type=int, default=1000, help="Occupancy window size.")
    args = parse_plotting_args(parser)

    df = load_data(args.experiment_path)
    df = filter_by_alg_aperture(df, args.filter_alg_apertures)

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    env = df["env"][0]
    biome_mapping = get_biome_mapping(env)

    if "TwoBiome" in env:
        biome_colors = TWO_BIOME_COLORS
        biome_order = ["Morel", "Oyster", "Neither"]
    elif "Weather" in env:
        biome_colors = WEATHER_BIOME_COLORS
        biome_order = ["Hot", "Cold", "Neither"]
    else:
        raise ValueError(f"Unknown biome colors for environment: {env}")

    # Filter to the last frame for the given sample type
    if args.sample_type == "end":
        df = df.filter(pl.col('frame') == pl.col('frame').max().over(['alg', 'aperture', 'seed']))
    else:
        df = df.filter(pl.col("sample_type") == args.sample_type)


    # Reshape data to be long-form for seaborn
    id_vars = ["alg", "aperture", "seed"]
    value_vars = [f"biome_{b}_occupancy_{args.window}" for b in biome_mapping.keys()]
    # Check which value_vars are actually in the dataframe
    value_vars = [v for v in value_vars if v in df.columns]

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, variable_name="metric", value_name="occupancy")

    long_df = long_df.with_columns(
        pl.col("metric").str.extract(r"biome_(-?\d+)_.*", 1).cast(pl.Int64).replace(biome_mapping).alias("biome")
    )

    # Create a combined label for the x-axis
    long_df = long_df.with_columns(
        (pl.col("alg").replace(LABEL_MAP).fill_null(pl.col("alg")) + pl.lit(" (A:") + pl.col("aperture").cast(pl.Utf8) + pl.lit(")")).alias("config_label")
    )


    # Plotting
    fig, ax = plt.subplots(layout="constrained")

    sns.barplot(
        data=long_df,
        x="config_label",
        y="occupancy",
        hue="biome",
        hue_order=biome_order,
        palette=biome_colors,
        ax=ax,
        errorbar=("ci", 95),
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Proportion of Time in Biome")
    ax.set_title(f"{env} - Biome Occupancy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    despine(ax)
    ax.legend(title="Biome", frameon=False)

    plot_name = args.plot_name or f"{env}_grouped_biome_bar"
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


if __name__ == "__main__":
    main()
