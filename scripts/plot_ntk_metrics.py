import sys
from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent.parent  # Go up from scripts/ to repo root
sys.path.insert(0, str(ROOT / "src"))

from utils.results import read_metrics_from_data

# Correct path to your results
data_path = ROOT / "results/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN/data"

print(f"Data path: {data_path}")
print(f"Data path exists: {data_path.exists()}")

# Load the metrics - first load all without filtering
df = read_metrics_from_data(
    data_path,
    metrics=None,  # Load all metrics first
    run_ids=None,
    sample=None,
).collect()

print(f"Data shape: {df.shape}")

# Check if our metrics exist
for metric in ["churn_norm", "ntk_rank", "ntk_cond"]:
    if metric in df.columns:
        non_nan = df[metric].drop_nulls().len()
        print(f"{metric}: {non_nan} non-NaN values out of {df.shape[0]}")
        # Show some sample values
        print(f"  Sample values: {df[metric].head(5).to_list()}")
        print(f"  Min: {df[metric].min()}, Max: {df[metric].max()}, Mean: {df[metric].mean()}")
    else:
        print(f"{metric}: NOT FOUND in data")

# Filter to non-NaN values
df_churn = df.filter(pl.col("churn_norm").is_not_null())
df_rank = df.filter(pl.col("ntk_rank").is_not_null())
df_cond = df.filter(pl.col("ntk_cond").is_not_null())

if len(df_churn) == 0 and len(df_rank) == 0 and len(df_cond) == 0:
    print("\nWARNING: No non-NaN metric values found!")
    print("The metrics may not have been computed during training.")
    exit(1)

# Group by frame and compute mean across seeds
if len(df_churn) > 0:
    churn = df_churn.group_by("frame").agg(pl.col("churn_norm").mean()).sort("frame")
else:
    churn = None

if len(df_rank) > 0:
    rank = df_rank.group_by("frame").agg(pl.col("ntk_rank").mean()).sort("frame")
else:
    rank = None

if len(df_cond) > 0:
    cond = df_cond.group_by("frame").agg(pl.col("ntk_cond").mean()).sort("frame")
else:
    cond = None

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

if churn is not None:
    axes[0].plot(churn["frame"], churn["churn_norm"], marker="o")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Churn Norm")
axes[0].set_title("Churn Over Time")

if rank is not None:
    axes[1].plot(rank["frame"], rank["ntk_rank"], marker="o")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("NTK Rank")
axes[1].set_title("NTK Rank Over Time")

if cond is not None:
    axes[2].plot(cond["frame"], cond["ntk_cond"], marker="o")
axes[2].set_xlabel("Step")
axes[2].set_ylabel("NTK Condition Number")
axes[2].set_title("NTK Condition Number Over Time")

plt.tight_layout()
plt.savefig("ntk_metrics.png")
plt.show()
