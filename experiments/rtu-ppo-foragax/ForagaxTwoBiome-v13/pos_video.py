import os
import sys
import polars as pl
import tol_colors as tc
# sys.path.append(os.getcwd() + "/src")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np
import re
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

colorset = tc.colorsets["muted"]

PALETTE = [
    colorset.rose,
    colorset.indigo,
    # colorset.sand,
    # colorset.cyan,
    colorset.teal,
    colorset.olive,
    colorset.purple,
    colorset.wine,
    colorset.green,
]

# Linestyles to distinguish families
LINESTYLES = {
    "RealTimeActorCriticMLP": "-",
    "ActorCriticMLP":        "-", 
    "Random": ":",                
}

# LINESTYLES = {
#     "RealTimeActorCriticConv-3": "-",
#     "RealTimeActorCriticConvEmb-3": "--",
#     "RealTimeActorCriticConvEmbNE-3": "-.",
#     "RealTimeActorCriticConvNE-3": ":",
#     "RealTimeActorCriticConvPooling-3": (0, (3, 1, 1, 1)),
#     "RealTimeActorCriticConvPoolingNE-3": (0, (5, 1)),
#     "RealTimeActorCriticMLP-3": (0, (1, 2)),
#     "RealTimeActorCriticMLPNE-3": (0, (3, 5, 1, 5)),
#     "Random": (0, (1, 1)),
# }

SINGLE = {
    "Random",
    "Search-Oracle",
    "Search-Nearest",
    "Search-Brown-Avoid-Green"
}

# Helper: strip optional "1M" token (with or without leading separator) so
# color is shared between 1M and non-1M variants

def base_without_1m(name: str) -> str:
    return re.sub(r"[-_]?1M", "", name)

if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel, metrics=["pos"])
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    env = Path(__file__).resolve().parent.name
    baselines_results = []

    # Collect sub-results grouped by FOV (aperture) so we can build a grid
    by_aperture = {}
    for key, sub_results in sorted(
        results.groupby_directory(level=3), key=lambda x: x[0]
    ):
        # level=3 directory is either an aperture like "9"/"15" or the literal "baselines"
        if str(key).lower() == "baselines":
            baselines_results = sub_results
            continue
        aperture = int(str(key))
        by_aperture.setdefault((env, aperture), []).extend(sub_results)

    # Determine unique apertures for this env (assumes a single env; if multiple, we create per-env figures)
    # Group by env first
    env_to_apertures = {}
    for (env, aperture), subs in by_aperture.items():
        env_to_apertures.setdefault(env, []).append(aperture)


    # --- Build a 2x3 figure for this env with specific comparisons ---
    fig, axes = plt.subplots(1, 3, squeeze=False, sharey=True, figsize=(15, 8))

    # Convenience mapping from (env, aperture) to list of results already exists: by_aperture
    # Helpers to identify algorithm families and variants
    def is_rtu(name: str) -> bool:
        return name.startswith("RealTimeActorCriticMLP")

    def is_ppo(name: str) -> bool:
        return name.startswith("ActorCriticMLP")

    def variant_token(name: str) -> str:
        # Return '', '1M', or '5M' based on filename tokens
        if "5M" in name:
            return "5M"
        if "1M" in name:
            return "1M"
        return ""

    def pretty_label(raw_alg: str, aperture: int | None, variant: str) -> str:
        # Map internal names to pretty labels requested in the prompt
        if raw_alg.startswith("Search-"):
            return raw_alg
        fam = "RTU-PPO" if is_rtu(raw_alg) else ("PPO" if is_ppo(raw_alg) else raw_alg)
        fov = f"FOV{aperture}" if aperture is not None else ""
        if variant == "1M":
            suffix = " frozen 1m"
        elif variant == "5M":
            suffix = " frozen 5m"
        else:
            suffix = ""
        return f"{fam} {fov}{suffix}".strip()

    # Line styles by variant: base solid, 1M dashed, 5M dotted
    def linestyle_for_variant(variant: str) -> str:
        return "-" if variant == "" else ("--" if variant == "1M" else ":")

    # Build color assignments so that base and its frozen variants share a color per family+FOV
    PALETTE_CYCLE = PALETTE  # reuse global palette
    color_index = 0
    base_color_map = {}

    def color_for(base_key: str) -> str:
        global color_index
        if base_key not in base_color_map:
            base_color_map[base_key] = PALETTE_CYCLE[color_index % len(PALETTE_CYCLE)]
            color_index += 1
        return base_color_map[base_key]

    # Utility to find and plot a single selection
    def plot_selection(ax, family: str, aperture: int | None, desired_variant: str | None, require_l2: bool=False, color_index:int=0):
        # family in {"SINGLE:Search-Oracle", "SINGLE:Search-Nearest", "RTU", "PPO"}
        if family.startswith("SINGLE:"):
            single_name = family.split(":", 1)[1]
            # Search SINGLE results across baselines_results; these are aperture-agnostic
            found = None
            for ar in baselines_results:
                if ar.filename == single_name:
                    found = ar
                    break
            if not found:
                return
            ar = found
            print(ar.filename)
            seed = 0
            df = ar.load(sample=int(sys.maxsize), start=9_999_000, end=10_000_000)
            if df is None:
                return
            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}
            seed_df = df.filter(pl.col('seed') == seed)
            xs, ys = extract_learning_curves(seed_df, hyper_vals=hyper_vals, metric="pos")
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            print("pos array shape:", ys.shape)

            # Expecting ys to be (trials, T, 2) or (1, T, 2) or (T, 2).
            # Normalize to pos with shape (T, 2)
            pos = ys
            if pos.ndim == 3 and pos.shape[0] == 1:
                pos = pos[0]
            elif pos.ndim == 3 and pos.shape[0] > 1:
                # pick first trial by default; could be extended to animate multiple trials
                pos = pos[0]
            elif pos.ndim == 2 and pos.shape[1] == 2:
                pos = pos
            else:
                print("Unexpected pos shape:", pos.shape)
                return

            # Create a simple animation of the 2D trajectory over time and save to MP4
            import matplotlib.animation as animation
            from pathlib import Path as _Path
            out_dir = _Path(__file__).resolve().parent / "plots" / "videos"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{single_name}_pos.mp4"

            fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
            ax_anim.set_title(f"{single_name} — trajectory")
            ax_anim.set_xlabel("x")
            ax_anim.set_ylabel("y")

            # set limits with a margin
            x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
            y_min, y_max = np.min(pos[:, 1]), np.max(pos[:, 1])
            x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
            y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
            ax_anim.set_xlim(x_min - x_margin, x_max + x_margin)
            ax_anim.set_ylim(y_min - y_margin, y_max + y_margin)

            # plot objects: trajectory line and moving point
            line, = ax_anim.plot([], [], linewidth=1.5)
            point, = ax_anim.plot([], [], marker='o', markersize=6)

            T = pos.shape[0]

            def init():
                line.set_data([], [])
                point.set_data([], [])
                return line, point

            def update(frame):
                # draw trajectory up to current frame and the current point
                # ensure we pass sequences (lists/arrays) to Line2D.set_data — matplotlib
                # raises `RuntimeError: x must be a sequence` if given plain scalars.
                line.set_data(pos[: frame + 1, 0], pos[: frame + 1, 1])
                x = pos[frame, 0]
                y = pos[frame, 1]
                # wrap scalars in lists so set_data always receives sequences
                point.set_data([x], [y])
                return line, point

            ani = animation.FuncAnimation(fig_anim, update, frames=T, init_func=init, blit=True, interval=20)

            # Try to save with FFMpegWriter if available, otherwise fall back to saving PNG frames and (optionally) requiring external assembly.
            try:
                writer = animation.FFMpegWriter(fps=30)
                ani.save(str(out_file), writer=writer)
                print(f"Saved animation to {out_file}")
            except Exception as e:
                print("FFmpeg writer failed or not available:", e)
                # fallback: write individual frames as PNGs into the out_dir/frames subfolder
                frames_dir = out_dir / f"{single_name}_frames"
                frames_dir.mkdir(exist_ok=True)
                for fidx in range(T):
                    update(fidx)
                    frame_path = frames_dir / f"frame_{fidx:04d}.png"
                    fig_anim.savefig(frame_path)
                print(f"Saved {T} PNG frames to {frames_dir}. You can assemble them into a video with ffmpeg:")
                print(f"ffmpeg -r 30 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {out_file}")

            plt.close(fig_anim)

            # return after making the video (don't plot this trajectory into the summary figure)
            return

        # Family selections (RTU or PPO) for a specific aperture and variant
        sub_results = by_aperture.get((env, aperture), []) if aperture is not None else []
        # desired_variant in {None (any), "" (base), "1M", "5M"}
        for ar in sub_results:
            name = ar.filename
            # Exclude any agent with "world" in its name
            if "world" in name.lower():
                continue
            if require_l2 and '-l2' not in name:
                continue
            if not require_l2 and '-l2' in name:
                continue
            if family == "RTU" and not is_rtu(name):
                continue
            if family == "PPO" and not is_ppo(name):
                continue
            vt = variant_token(name)
            if desired_variant is None:
                pass  # accept any
            elif desired_variant == "":
                if vt != "":
                    continue
            else:
                if vt != desired_variant:
                    continue
            print(name)
            # We have a match; plot it and return
            df = ar.load(sample=int(sys.maxsize), start=9_999_000, end=10_000_000)
            if df is None:
                return
            seed = 0
            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}
            seed_df = df.filter(pl.col('seed') == seed)
            xs, ys = extract_learning_curves(seed_df, hyper_vals=hyper_vals, metric="pos")
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            print("pos array shape:", ys.shape)

            # Expecting ys to be (trials, T, 2) or (1, T, 2) or (T, 2).
            # Normalize to pos with shape (T, 2)
            pos = ys
            if pos.ndim == 3 and pos.shape[0] == 1:
                pos = pos[0]
            elif pos.ndim == 3 and pos.shape[0] > 1:
                # pick first trial by default; could be extended to animate multiple trials
                pos = pos[0]
            elif pos.ndim == 2 and pos.shape[1] == 2:
                pos = pos
            else:
                print("Unexpected pos shape:", pos.shape)
                return

            # Create a simple animation of the 2D trajectory over time and save to MP4
            import matplotlib.animation as animation
            from pathlib import Path as _Path
            out_dir = _Path(__file__).resolve().parent / "plots" / "videos"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{name}_{aperture}_pos.mp4"

            fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
            ax_anim.set_title(f"{name} {aperture}— trajectory")
            ax_anim.set_xlabel("x")
            ax_anim.set_ylabel("y")

            # set limits with a margin
            x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
            y_min, y_max = np.min(pos[:, 1]), np.max(pos[:, 1])
            x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
            y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
            ax_anim.set_xlim(x_min - x_margin, x_max + x_margin)
            ax_anim.set_ylim(y_min - y_margin, y_max + y_margin)

            # plot objects: trajectory line and moving point
            line, = ax_anim.plot([], [], linewidth=1.5)
            point, = ax_anim.plot([], [], marker='o', markersize=6)

            T = pos.shape[0]

            def init():
                line.set_data([], [])
                point.set_data([], [])
                return line, point

            def update(frame):
                # draw trajectory up to current frame and the current point
                # ensure we pass sequences (lists/arrays) to Line2D.set_data — matplotlib
                # raises `RuntimeError: x must be a sequence` if given plain scalars.
                line.set_data(pos[: frame + 1, 0], pos[: frame + 1, 1])
                x = pos[frame, 0]
                y = pos[frame, 1]
                # wrap scalars in lists so set_data always receives sequences
                point.set_data([x], [y])
                return line, point

            ani = animation.FuncAnimation(fig_anim, update, frames=T, init_func=init, blit=True, interval=20)

            # Try to save with FFMpegWriter if available, otherwise fall back to saving PNG frames and (optionally) requiring external assembly.
            try:
                writer = animation.FFMpegWriter(fps=30)
                ani.save(str(out_file), writer=writer)
                print(f"Saved animation to {out_file}")
            except Exception as e:
                print("FFmpeg writer failed or not available:", e)
                # fallback: write individual frames as PNGs into the out_dir/frames subfolder
                frames_dir = out_dir / f"{name}_frames"
                frames_dir.mkdir(exist_ok=True)
                for fidx in range(T):
                    update(fidx)
                    frame_path = frames_dir / f"frame_{fidx:04d}.png"
                    fig_anim.savefig(frame_path)
                print(f"Saved {T} PNG frames to {frames_dir}. You can assemble them into a video with ffmpeg:")
                print(f"ffmpeg -r 30 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {out_file}")

            plt.close(fig_anim)

            # return after making the video (don't plot this trajectory into the summary figure)
            return
        # If no match found, just return silently
        return

    # ---------------- Subplot 1 ----------------
    ax = axes[0][0]
    # Two search baselines
    plot_selection(ax, "SINGLE:Search-Oracle", None, "")
    plot_selection(ax, "SINGLE:Search-Nearest", None, "")
    plot_selection(ax, "SINGLE:Search-Brown-Avoid-Green", None, "")
    # RTU/PPO at FOV 9 and 15 (base, non-frozen)
    plot_selection(ax, "RTU", 9, "", color_index = 0)
    # plot_selection(ax, "PPO", 9, "", color_index = 1)
    plot_selection(ax, "RTU", 15, "", color_index = 2)
    # plot_selection(ax, "PPO", 15, "", color_index = 3)
    assert False
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"{env} — Baselines & FOV 9/15")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 2 ----------------
    # ax = axes[0][1]
    # # RTU-PPO FOV9: base, frozen 1m, frozen 5m
    # plot_selection(ax, "RTU", 9, "", color_index=0)
    # plot_selection(ax, "RTU", 9, "1M", color_index=1)
    # plot_selection(ax, "RTU", 9, "5M", color_index=2)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — RTU-PPO FOV 9: base vs frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 3 ----------------
    # ax = axes[0][2]
    # # PPO FOV9: frozen 1m, frozen 5m
    # plot_selection(ax, "PPO", 9, "", color_index = 0)
    # plot_selection(ax, "PPO", 9, "1M", color_index = 1)
    # plot_selection(ax, "PPO", 9, "5M", color_index = 2)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — PPO FOV 9: frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 4 ----------------
    # ax = axes[1][0]
    # # Two search baselines -l2
    # plot_selection(ax, "SINGLE:Search-Oracle", None, "")
    # plot_selection(ax, "SINGLE:Search-Nearest", None, "")
    # plot_selection(ax, "SINGLE:Search-Brown-Avoid-Green", None, "")
    # # RTU/PPO at FOV 9 and 15 (base, non-frozen) -l2
    # plot_selection(ax, "RTU", 9, "", require_l2=True)
    # plot_selection(ax, "PPO", 9, "", require_l2=True)
    # plot_selection(ax, "RTU", 15, "", require_l2=True)
    # plot_selection(ax, "PPO", 15, "", require_l2=True)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_ylabel("Average Reward")
    # ax.set_title(f"{env} — Baselines-l2 & FOV 9/15")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 5 ----------------
    # ax = axes[1][1]
    # # RTU-PPO FOV9: base-l2, frozen 1m-l2, frozen 5m-l2
    # plot_selection(ax, "RTU", 9, "", require_l2=True)
    # plot_selection(ax, "RTU", 9, "1M", require_l2=True)
    # plot_selection(ax, "RTU", 9, "5M", require_l2=True)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — RTU-PPO FOV 9-l2: base vs frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 6 ----------------
    # ax = axes[1][2]
    # # PPO FOV9: frozen 1m-l2, frozen 5m-l2
    # plot_selection(ax, "PPO", 9, "", require_l2=True)
    # plot_selection(ax, "PPO", 9, "1M", require_l2=True)
    # plot_selection(ax, "PPO", 9, "5M", require_l2=True)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — PPO FOV 9-l2: frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # Save one figure per env containing the 3 subplots
    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name=f"three_panel",
        save_type="pdf",
        f=fig,
        width=5,
        height_ratio=1/5,
    )

    plt.close(fig)
