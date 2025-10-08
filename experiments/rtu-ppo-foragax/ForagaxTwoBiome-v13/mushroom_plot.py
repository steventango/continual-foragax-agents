import os
import sys
import tol_colors as tc
# sys.path.append(os.getcwd() + "/src")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib as mpl
from matplotlib import pyplot as plt, patheffects
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

# Helper: interpolate y at a given x for a Line2D
def _interp_y_at_x(line, x):
    xdata = np.asarray(line.get_xdata(), dtype=float)
    ydata = np.asarray(line.get_ydata(), dtype=float)
    # keep finite points only
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[mask]
    ydata = ydata[mask]
    if xdata.size < 2:
        return None
    # ensure increasing x for interp
    idx = np.argsort(xdata)
    xdata = xdata[idx]
    ydata = ydata[idx]
    if x < xdata[0] or x > xdata[-1]:
        return None
    return float(np.interp(x, xdata, ydata))

# Helper function to label lines directly at an interior location
def label_line(ax, line, label, x_frac=0.85, x_offset_points=5, y_offset_points=None):
    """
    Place a text label near the right side but inside the axes (not on the edge).
    - x_frac: fraction between current x-limits where to place the label (0<x_frac<1).
    - Offsets are in display points to avoid scaling with data limits.
    If y_offset_points=None, automatically chooses a vertical offset to avoid covering the line.
    Additionally, choose an anchor (xy) that is far from anchors already used on this axes
    to reduce overlapping/stacking. Distance check is done in display pixels.
    """
    if not label or label.startswith("_"):  # ignore no-label lines
        return

    # Create storage for previously used anchor points on this axes (in display coords)
    if not hasattr(ax, "_label_xy_used"):
        ax._label_xy_used = []  # list of np.array([xp, yp]) in pixels

    # choose initial interior x based on current view limits
    xmin, xmax = ax.get_xlim()

    def interp_y_safe(x):
        yval = _interp_y_at_x(line, x)
        if yval is None:
            # fallback: try last finite point within view
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)
            mask = np.isfinite(xdata) & np.isfinite(ydata) & (xdata >= xmin) & (xdata <= xmax)
            if not np.any(mask):
                return None, None
            return float(xdata[mask][-1]), float(ydata[mask][-1])
        return x, float(yval)

    # Generate candidate x positions by spreading around requested x_frac
    # and keeping them inside [0.10, 0.70] of the current range
    spread_fracs = [0.0, -0.07, 0.07, -0.14, 0.14, -0.21, 0.21]
    cand_fracs = []
    for df in spread_fracs:
        cf = x_frac + df
        cf = max(0.10, min(0.70, cf))
        if cf not in cand_fracs:
            cand_fracs.append(cf)

    candidates = []  # list of tuples (xdata, ydata, min_dist_pixels)
    for cf in cand_fracs:
        x_try = xmin + cf * (xmax - xmin)
        x_i, y_i = interp_y_safe(x_try)
        if x_i is None or y_i is None:
            continue
        # distance to existing anchors in pixels
        xp, yp = ax.transData.transform((x_i, y_i))
        if ax._label_xy_used:
            d = min(np.hypot(xp - p[0], yp - p[1]) for p in ax._label_xy_used)
        else:
            d = float("inf")
        candidates.append((x_i, y_i, d))

    if not candidates:
        return

    # Prefer the candidate that maximizes the minimum distance to existing anchors
    x_sel, y_sel, d_sel = max(candidates, key=lambda t: t[2])

    # If still too close (< 25 px), try nudging vertically by sampling a few y-offsets in data coords
    # Convert ~12, 24 points to data units using inverse of a transform step
    MIN_D = 25.0
    if d_sel < MIN_D:
        # approximate vertical data delta for 12 points
        p0 = ax.transData.inverted().transform((0, 0))
        p12 = ax.transData.inverted().transform((0, 12))
        dy_data = abs(p12[1] - p0[1])
        for mult in [1, -1, 2, -2]:
            y_try = y_sel + mult * dy_data
            xp, yp = ax.transData.transform((x_sel, y_try))
            if ax._label_xy_used:
                d = min(np.hypot(xp - p[0], yp - p[1]) for p in ax._label_xy_used)
            else:
                d = float("inf")
            if d > d_sel:
                y_sel, d_sel = y_try, d
                if d_sel >= MIN_D:
                    break

    # record the chosen anchor in display coordinates
    ax._label_xy_used.append(np.array(ax.transData.transform((x_sel, y_sel))))

    # Decide horizontal alignment and x-offset so text never crosses the right boundary
    # and never extends farther right than the line's last x-position
    # Compute the display x of the anchor and compare against the axes' and line's right edges
    x_display_right = ax.transAxes.transform((1.0, 0.0))[0]
    xp_anchor, yp_anchor = ax.transData.transform((x_sel, y_sel))

    # Find the line's last finite x within data
    xdata = np.asarray(line.get_xdata(), dtype=float)
    ydata = np.asarray(line.get_ydata(), dtype=float)
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    if np.any(mask):
        x_last = float(np.max(xdata[mask]))
        # y at last x: take last corresponding y of that x (approx via mask == x_last)
        # fallback to interpolation if duplicates/unsorted
        same = np.isclose(xdata, x_last) & mask
        if np.any(same):
            y_last = float(ydata[same][-1])
        else:
            y_last = _interp_y_at_x(line, x_last) or float(ydata[mask][-1])
    else:
        x_last = ax.get_xlim()[1]
        y_last = y_sel

    xp_line_end = ax.transData.transform((x_last, y_last))[0]

    # The absolute allowable right edge for the label text (in pixels)
    SAFE_PAD = 2.0
    allowable_right = min(x_display_right, xp_line_end) - SAFE_PAD

    # Measure label text width in pixels using a temporary text object
    fig = ax.figure
    try:
        fig.canvas.draw()  # ensure a renderer exists
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    text_w = 0.0
    if renderer is not None:
        tmp = ax.text(0, 0, label)
        bbox = tmp.get_window_extent(renderer=renderer)
        text_w = bbox.width
        tmp.remove()

    # Default placement: to the right of anchor
    ha_val = "left"
    x_text_off = x_offset_points

    # If placing to the right would exceed allowable_right, right-align and pin the right edge
    if xp_anchor + x_offset_points + text_w > allowable_right:
        ha_val = "right"
        # Set offset so that the text's right edge equals allowable_right
        x_text_off = allowable_right - xp_anchor

    # Auto y-offset to avoid covering the line if not provided
    if y_offset_points is None:
        # estimate local slope using central difference around the selected x
        xmin_view, xmax_view = ax.get_xlim()
        dx = max(1e-9, 0.01 * (xmax_view - xmin_view))
        y_left = _interp_y_at_x(line, x_sel - dx)
        y_right = _interp_y_at_x(line, x_sel + dx)
        if y_left is None or y_right is None:
            slope = 0.0
        else:
            slope = (y_right - y_left) / (2.0 * dx)
        # Choose vertical offset direction: when text is to the right (ha left),
        # put text below the line for positive slope and above for negative slope.
        # When text is to the left (ha right), flip the rule.
        base_mag = 12.0  # points
        if ha_val == "left":
            y_offset_points = -base_mag if slope > 0 else base_mag
        else:
            y_offset_points = base_mag if slope > 0 else -base_mag

    color = line.get_color()
    ax.annotate(
        label,
        xy=(x_sel, y_sel),
        xycoords="data",
        xytext=(x_text_off, y_offset_points),
        textcoords="offset points",
        ha=ha_val,
        va="center",
        color=color,
        path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
        clip_on=False,
    )

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
    results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
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
    fig, axes = plt.subplots(1, 1, squeeze=False, sharey=True, figsize=(15, 8))

    # add a small right margin so text outside the axes isn't cut off in saves
    for row in axes:
        for ax in row:
            ax.margins(x=0.01, y=0.02)

    # Convenience mapping from (env, aperture) to list of results already exists: by_aperture
    # Helpers to identify algorithm families and variants
    def is_rtu(name: str) -> bool:
        return name.startswith("RealTimeActorCriticMLP")

    def is_ppo(name: str) -> bool:
        return name.startswith("ActorCriticMLP")
    
    def is_dqn(name: str) -> bool:
        return name.startswith("DQN")

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
        fam = "RTU-PPO" if is_rtu(raw_alg) else ("PPO" if is_ppo(raw_alg) else ("DQN" if is_dqn(raw_alg) else raw_alg))
        fov = f"FOV{aperture}" if aperture is not None else ""
        if variant == "1M":
            suffix = " frozen 1m"
        elif variant == "5M":
            suffix = " frozen 5m"
        else:
            suffix = ""
        if "l2" in raw_alg:
            suffix = " L2 " + suffix
        if "world" in raw_alg:
            suffix = " True State " + suffix
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
    def plot_selection(ax, family: str, aperture: int | None, desired_variant: str | None, require_l2: bool=False, require_world=False, color_index:int=0):
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
            df = ar.load(end=10_000_000)
            if df is None:
                return
            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}
            xs, ys = extract_learning_curves(df, hyper_vals=hyper_vals, metric="ewm_reward")
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            assert np.all(np.isclose(xs[0], xs))
            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0), y=ys, statistic=Statistic.mean, iterations=10000
            )
            base_key = f"SINGLE-{single_name}"
            variant_color = PALETTE_CYCLE[(color_index) % len(PALETTE_CYCLE)]
            label = pretty_label(single_name, None, "")
            ax.plot(xs[0], res.sample_stat, label=label, color=variant_color, linewidth=1.0)
            if len(ys) >= 5:
                ax.fill_between(xs[0], res.ci[0], res.ci[1], color=variant_color, alpha=0.1)
            else:
                for y in ys:
                    ax.plot(xs[0], y, color=variant_color, linewidth=0.2)
            return

        # Family selections (RTU or PPO) for a specific aperture and variant
        sub_results = by_aperture.get((env, aperture), []) if aperture is not None else []
        # desired_variant in {None (any), "" (base), "1M", "5M"}
        for ar in sub_results:
            name = ar.filename
            if require_world and 'world' not in name:
                continue
            if not require_world and 'world' in name:
                continue
            if require_l2 and '-l2' not in name:
                continue
            if not require_l2 and '-l2' in name:
                continue
            if family == "RTU" and not is_rtu(name):
                continue
            if family == "PPO" and not is_ppo(name):
                continue
            if family == "DQN" and not is_dqn(name):
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
            df = ar.load(end=10_000_000)
            if df is None:
                return
            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}
            xs, ys = extract_learning_curves(df, hyper_vals=hyper_vals, metric="ewm_reward")
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            assert np.all(np.isclose(xs[0], xs))
            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0), y=ys, statistic=Statistic.mean, iterations=10000
            )
            base_key = f"{family}-FOV{aperture}"
            variant_color = PALETTE_CYCLE[(color_index) % len(PALETTE_CYCLE)]
            label = pretty_label(name, aperture, vt)
            ax.plot(xs[0], res.sample_stat, label=label, color=variant_color, linewidth=1.0)
            if len(ys) >= 5:
                ax.fill_between(xs[0], res.ci[0], res.ci[1], color=variant_color, alpha=0.1)
            else:
                for y in ys:
                    ax.plot(xs[0], y, color=variant_color, linewidth=0.2)
            return
        # If no match found, just return silently
        return

    # ---------------- Subplot 1 ----------------
    ax = axes[0][0]
    # Two search baselines
    plot_selection(ax, "SINGLE:Search-Oracle", None, "", color_index = 0)
    plot_selection(ax, "SINGLE:Search-Nearest", None, "", color_index = 1)
    plot_selection(ax, "SINGLE:Search-Brown-Avoid-Green", None, "", color_index = 1)
    # # RTU/PPO at FOV 9 and 15 (base, non-frozen)
    # plot_selection(ax, "RTU", 9, "", color_index = 0)
    # plot_selection(ax, "PPO", 9, "", color_index = 1)
    plot_selection(ax, "DQN", 15, "", color_index = 2)
    plot_selection(ax, "DQN", 15, "", require_world=True, color_index = 3)
    # plot_selection(ax, "RTU", 15, "", color_index = 2)
    plot_selection(ax, "PPO", 15, "", color_index = 4)
    plot_selection(ax, "PPO", 15, "", require_world=True,  color_index = 5)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Average Reward")
    # ax.set_title(f"{env} — Baselines & FOV 9/15")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Label primary lines directly (linewidth >= 1) — scatter labels across the x-axis
    primary_lines = [ln for ln in ax.get_lines() if ln.get_linewidth() >= 1.0]
    if primary_lines:
        # Evenly spread label positions between 5% and 55% of the current x-range
        xmin, xmax = ax.get_xlim()
        x_fracs = np.linspace(0.05, 0.55, len(primary_lines))
        for line, xf in zip(primary_lines, x_fracs):
            label_line(
                ax,
                line,
                line.get_label(),
                x_frac=float(xf),
                x_offset_points=4,
                y_offset_points=None,
            )

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
    # # Label primary lines directly (linewidth >= 1)
    # for i, line in enumerate([ln for ln in ax.get_lines() if ln.get_linewidth() >= 1.0]):
    #     # stagger y-offset a bit to reduce overlap
    #     label_line(ax, line, line.get_label(), x_frac=0.85, y_offset_points=(i % 3 - 1) * 6)

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
    # # Label primary lines directly (linewidth >= 1)
    # for i, line in enumerate([ln for ln in ax.get_lines() if ln.get_linewidth() >= 1.0]):
    #     # stagger y-offset a bit to reduce overlap
    #     label_line(ax, line, line.get_label(), x_frac=0.85, y_offset_points=(i % 3 - 1) * 6)

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
    # improve layout to reduce trimming issues
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.subplots_adjust(top=0.995, right=0.995)
    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name=f"baselines_ppo_dqn_and_true_states",
        save_type="pdf",
        f=fig,
        width=1,
        height_ratio=1,
    )

    plt.close(fig)
