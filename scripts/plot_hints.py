import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rcParams["font.family"] = "sans-serif"

files = {
    "Search-Oracle": "/home/steven/Github/continual-foragax-agents/results/E136-big/foragax/ForagaxBig-v4/Baselines/Search-Oracle/data/0.npz",
    "DQN_LN": "/home/steven/Github/continual-foragax-agents/results/E136-big/foragax/ForagaxBig-v4/9/DQN_LN/data/0.npz",
}

colors = ["#3498db", "#e74c3c"]
n_algs = len(files)

fig, axes = plt.subplots(
    n_algs + 1,
    1,
    figsize=(14, 4 * (n_algs + 1)),
    gridspec_kw={"height_ratios": [1] * n_algs + [1]},
)

# Share x-axis only among the scatter/step subplots, not the bar chart
for i in range(1, n_algs):
    axes[i].sharex(axes[0])

# --- Per-algorithm scatter plots ---
all_hints = {}
for ax, (label, path), color in zip(axes[:n_algs], files.items(), colors):
    data = np.load(path)
    hints = data["hint"][:1000]
    all_hints[label] = hints

    active_mask = hints != -1
    active_steps = np.where(active_mask)[0]
    inactive_steps = np.where(~active_mask)[0]

    # Subsample for performance / visual clarity
    max_inactive = 50000
    if len(inactive_steps) > max_inactive:
        idx = np.linspace(0, len(inactive_steps) - 1, max_inactive, dtype=int)
        inactive_steps = inactive_steps[idx]

    active_count = int(active_mask.sum())
    ax.scatter(inactive_steps, hints[inactive_steps], s=1, alpha=0.15, color="gray")
    ax.scatter(
        active_steps,
        hints[active_steps],
        s=3,
        alpha=0.7,
        color=color,
        label=label,
        linewidths=0,
    )
    ax.set_ylabel("Hint ID", fontsize=14)
    ax.set_title(
        f"{label}  ({active_count:,} / {len(hints):,} steps have a hint)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)


# --- Hint frequency subplot: count of each hint value (0, 1, 2, 3) ---
ax_freq = axes[-1]
hint_values = [0, 1, 2, 3]
bar_width = 0.35
x = np.arange(len(hint_values))

for i, ((label, hints), color) in enumerate(zip(all_hints.items(), colors)):
    counts = [int(np.sum(hints == v)) for v in hint_values]
    ax_freq.bar(
        x + i * bar_width, counts, bar_width, label=label, color=color, alpha=0.8
    )

ax_freq.set_xticks(x + bar_width / 2)
ax_freq.set_xticklabels([str(v) for v in hint_values])
ax_freq.set_xlabel("Hint ID", fontsize=14)
ax_freq.set_ylabel("Count", fontsize=14)
ax_freq.set_title("Hint Frequency (excluding -1)", fontsize=14, fontweight="bold")
ax_freq.legend(loc="upper right", fontsize=12)
ax_freq.grid(True, linestyle="--", alpha=0.5, axis="y")


fig.suptitle("Hints Over Time (-1 = no hint)", fontsize=18, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("hint_comparison.png", dpi=150, bbox_inches="tight")
print("Plot saved to hint_comparison.png")
