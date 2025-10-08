import argparse
import json
import math
import os
import sys
from pathlib import Path

from annotate_plot import annotate_plot

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from plotting_utils import (
    PlottingArgumentParser,
    despine,
    load_data,
    parse_plotting_args,
    save_plot,
)


def main():
    parser = PlottingArgumentParser(description="Plot object trace metrics.")
    parser.add_argument(
        "--trace-exponent",
        type=int,
        required=True,
        help="Exponent for the object trace metric.",
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="every",
        help="Sample type to filter from the data.",
    )
    parser.add_argument(
        "--auto-label", action="store_true", help="Enable auto-labeling."
    )

    args = parse_plotting_args(parser)

    df = load_data(args.experiment_path)
    df = df.filter(pl.col("sample_type") == args.sample_type)

    if args.filter_algs:
        df = df.filter(pl.col("alg").is_in(args.filter_algs))

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    env = df["env"][0]

    if "TwoBiome" in env:
        object_mapping = {0: "Morel", 1: "Oyster"}
    elif "Weather" in env:
        object_mapping = {0: "Hot", 1: "Cold"}
    else:
        if "object_collected_id" in df.columns:
            object_mapping = {
                obj_id: f"Object {obj_id}"
                for obj_id in df["object_collected_id"].unique()
                if obj_id is not None
            }
        else:
            object_mapping = {}

    available_objects = sorted([k for k in object_mapping.keys()])

    trace_metrics = [
        f"object_trace_{obj}_{args.trace_exponent}" for obj in available_objects
    ]
    mushroom_names = [object_mapping[obj] for obj in available_objects]

    df_melted = df.melt(
        id_vars=["frame", "alg", "seed"],
        value_vars=trace_metrics,
        variable_name="metric",
        value_name="value",
    )

    metric_to_name = dict(zip(trace_metrics, mushroom_names, strict=True))
    df_melted = df_melted.with_columns(
        pl.col("metric").replace(metric_to_name).alias("Object")
    )

    g = sns.relplot(
        data=df_melted.to_pandas(),
        x="frame",
        y="value",
        hue="Object",
        col="alg",
        kind="line",
        errorbar=("ci", 95),
        col_wrap=min(len(df["alg"].unique()), 3),
        facet_kws=dict(sharey=True),
    )

    g.set_axis_labels("Time steps", "Trace Value")
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle(f"{env} - Object Traces (e={args.trace_exponent})", y=1.03)

    for ax in g.axes.flatten():
        despine(ax)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        if args.auto_label:
            annotate_plot(ax)

    if not args.auto_label:
        g.add_legend(title=None, frameon=False)

    plot_name = args.plot_name or f"{env}_object_trace_e{args.trace_exponent}"
    save_plot(g.fig, args.experiment_path, plot_name, args.save_type)

if __name__ == "__main__":
    main()
