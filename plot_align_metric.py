# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Global Parameters
n_steps = 500_000
noise_std = 1.0
ewm_noise_std = 0.5
downsample_step = 5000
alpha1 = 1e-3
alpha2 = 1e-5

# Time arrays
t_full = np.arange(n_steps)
t_half1 = np.arange(n_steps // 2)
t_half2 = np.arange(n_steps // 2, n_steps)


def shark_fin_flat_wave(t, period, offset=0):
    """
    Generates a wave that rises like a shark fin from 0 to 1 for the first half,
    and stays flat at 1 for the second half.
    """
    phase = ((t + offset) % period) / period
    first_half = phase < 0.5

    wave = np.zeros_like(t, dtype=float)

    # First half: Exponential increase exactly from 0 to 1
    norm_time = phase[first_half] * 2
    # Normalizing so the peak mathematically hits 1.0 before the flat line
    wave[first_half] = (1 - np.exp(-5 * norm_time)) / (1 - np.exp(-5))

    # Second half: Flat line at 1
    wave[~first_half] = 1.0

    return wave


def square_wave_0_1(t, period, offset=0):
    """
    Generates a square wave alternating between 0 (first half) and 1 (second half).
    """
    phase = ((t + offset) % period) / period
    wave = np.zeros_like(t, dtype=float)

    # First half of the period: Flat line at 0
    wave[phase < 0.5] = 0.0

    # Second half of the period: Flat line at 1
    wave[phase >= 0.5] = 1.0

    return wave


def generate_custom_data(n_curves, randomize_offset=False):
    """Generates the raw dataset for a given number of curves per group."""
    data = {}

    # Group 1: Shark Fin -> Flat 1 (Period 100k for the full 1M steps)
    for i in range(n_curves):
        offset = np.random.randint(0, 100_000) if randomize_offset else 0
        wave = shark_fin_flat_wave(t_full, 100_000, offset=offset)
        noise = np.random.normal(0, noise_std, n_steps)
        data[f"Group 1_Curve {i}"] = wave + noise

    # Group 2: Shark Fin -> Flat 1 (First 500k), then Square Wave 0 -> 1 (Next 500k)
    for i in range(n_curves):
        offset = np.random.randint(0, 100_000) if randomize_offset else 0
        wave_p1 = shark_fin_flat_wave(t_half1, 100_000, offset=offset)
        # Period is 100k (50k at 0, 50k at 1) starting at 500k
        wave_p2 = square_wave_0_1(t_half2, 100_000, offset=offset)
        wave = np.concatenate([wave_p1, wave_p2])

        noise = np.random.normal(0, noise_std, n_steps)
        data[f"Group 2_Curve {i}"] = wave + noise

    return pd.DataFrame(data)


def prep_for_seaborn(df, downsample):
    """Downsamples and reshapes dataframe into long format for Seaborn."""
    df_sub = df.iloc[::downsample].copy()
    df_sub["Step"] = df_sub.index
    df_melt = df_sub.melt(id_vars="Step", var_name="Curve_ID", value_name="Value")
    df_melt["Group"] = df_melt["Curve_ID"].apply(lambda x: x.split("_")[0])
    return df_melt


# 1. Generate & Process Data
def process_data(n_curves, randomize_offset=False):
    df_raw = generate_custom_data(n_curves, randomize_offset) + np.random.normal(
        0, ewm_noise_std
    )
    df_ewm_noise_1 = df_raw.ewm(alpha=alpha1, adjust=True).mean()
    df_ewm_noise_2 = df_raw.ewm(alpha=alpha2, adjust=True).mean()

    return {
        "raw": prep_for_seaborn(df_raw, downsample_step),
        "ewm_noise_1": prep_for_seaborn(df_ewm_noise_1, downsample_step),
        "ewm_noise_2": prep_for_seaborn(df_ewm_noise_2, downsample_step),
    }


data_5 = process_data(5)
data_30 = process_data(30)
data_5_off = process_data(5, randomize_offset=True)
data_30_off = process_data(30, randomize_offset=True)

# 2. Plotting Setup
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(5, 4, figsize=(40, 25), sharex=True, sharey="row")


def plot_col(col_idx, title_prefix, d, alpha, show_legend=False):
    ax_col = axes[:, col_idx]

    sns.lineplot(
        data=d["raw"],
        x="Step",
        y="Value",
        hue="Group",
        units="Curve_ID",
        estimator=None,
        alpha=alpha,
        ax=ax_col[0],
        legend=show_legend,
    )
    ax_col[0].set_title(f"{title_prefix}: Raw Noisy Mixed Waves")
    if show_legend:
        ax_col[0].legend(loc="upper right", bbox_to_anchor=(1.3, 1))

    sns.lineplot(
        data=d["ewm_noise_1"],
        x="Step",
        y="Value",
        hue="Group",
        units="Curve_ID",
        estimator=None,
        alpha=alpha,
        ax=ax_col[1],
        legend=False,
    )
    ax_col[1].set_title(f"{title_prefix}: Individual EWM ($\\alpha={alpha1}$)")

    sns.lineplot(
        data=d["ewm_noise_2"],
        x="Step",
        y="Value",
        hue="Group",
        units="Curve_ID",
        estimator=None,
        alpha=alpha,
        ax=ax_col[2],
        legend=False,
    )
    ax_col[2].set_title(f"{title_prefix}: Individual EWM ($\\alpha={alpha2}$)")

    sns.lineplot(
        data=d["ewm_noise_1"], x="Step", y="Value", hue="Group", ax=ax_col[3], legend=False
    )
    ax_col[3].set_title(f"{title_prefix}: EWM ($\\alpha={alpha1}$) with 95% CI")

    sns.lineplot(
        data=d["ewm_noise_2"], x="Step", y="Value", hue="Group", ax=ax_col[4], legend=False
    )
    ax_col[4].set_title(f"{title_prefix}: EWM ($\\alpha={alpha2}$) with 95% CI")
    ax_col[4].set_xlabel("Step")


plot_col(0, "5 Curves", data_5, alpha=0.4)
plot_col(1, "30 Curves", data_30, alpha=0.15, show_legend=True)
plot_col(2, "5 Curves Offset", data_5_off, alpha=0.4)
plot_col(3, "30 Curves Offset", data_30_off, alpha=0.15)

for ax in axes.flat:
    ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig("align_metric.png")
# %%
