"""
Visualize CNN activations from a DQN_LN_PConv agent alongside environment observations.

Generates a video with:
  - LEFT:  The 9×9 RGB observation (upscaled for clarity)
  - RIGHT: Conv activation feature maps for each convolutional layer

Usage:
    python scripts/visualize_cnn_activations.py <path_to_glue_state.pkl.xz> \\
        [--output <output_path.mp4>] [--steps 500] [--fps 10] [--policy]

Examples:
    # Random policy, 300 steps
    python scripts/visualize_cnn_activations.py \\
        checkpoints/results/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_PConv/0/glue_state.pkl.xz

    # Use learned greedy policy
    python scripts/visualize_cnn_activations.py \\
        checkpoints/results/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_PConv/0/glue_state.pkl.xz \\
        --policy --steps 500
"""

import argparse
import lzma
import os
import pickle

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── Loading & extraction ─────────────────────────────────────────────────


def load_glue_state(path: str):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def to_numpy(x):
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def extract_conv_params(params: dict) -> list[dict]:
    """
    Extract conv-layer weights from Haiku params.

    PConv2D ForagerNet structure:
        phi/~/phi       -> 1×1 pointwise  (w: 1,1,3,16  b: 16)
        phi/~/layer_norm                   (scale/offset: 16)
        phi/~/phi_1     -> 3×3 conv        (w: 3,3,16,16 b: 16)
        phi/~/layer_norm_1                 (scale/offset: 16)
    """
    phi_params = params["phi"]
    conv_modules = []
    ln_modules = {}

    for path in sorted(phi_params.keys()):
        mod = phi_params[path]
        if not hasattr(mod, "keys"):
            continue
        if "w" in mod and to_numpy(mod["w"]).ndim == 4:
            conv_modules.append((path, mod))
        if "scale" in mod and "offset" in mod:
            ln_modules[path] = mod

    layers = []
    for i, (path, mod) in enumerate(conv_modules):
        w = to_numpy(mod["w"])
        b = to_numpy(mod["b"]) if "b" in mod else None

        # pair with layer-norm by index convention
        if i == 0:
            ln_candidates = [
                k
                for k in ln_modules
                if "layer_norm" in k
                and not any(c.isdigit() for c in k.split("layer_norm")[-1])
            ]
        else:
            ln_candidates = [k for k in ln_modules if f"layer_norm_{i}" in k]

        ln_scale = ln_offset = None
        if ln_candidates:
            ln = ln_modules[ln_candidates[0]]
            ln_scale, ln_offset = to_numpy(ln["scale"]), to_numpy(ln["offset"])

        H, W, C_in, C_out = w.shape
        layers.append(
            dict(
                name=path,
                w=w,
                b=b,
                ln_scale=ln_scale,
                ln_offset=ln_offset,
                kernel_size=(H, W),
                c_in=C_in,
                c_out=C_out,
            )
        )
    return layers


# ── Forward helpers ──────────────────────────────────────────────────────


def apply_conv_layer(x: np.ndarray, layer: dict) -> np.ndarray:
    """Conv → (LayerNorm) → ReLU  on a single obs  (H,W,C) → (H',W',C_out)."""
    x_j = jnp.array(x)[None]  # (1,H,W,C)
    w = jnp.array(layer["w"])
    h = jax.lax.conv_general_dilated(
        x_j, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    if layer["b"] is not None:
        h = h + jnp.array(layer["b"])
    if layer["ln_scale"] is not None:
        s, o = jnp.array(layer["ln_scale"]), jnp.array(layer["ln_offset"])
        mu = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = (h - mu) / jnp.sqrt(var + 1e-5) * s + o
    h = jax.nn.relu(h)
    return np.asarray(h[0])


def get_activations(obs: np.ndarray, conv_layers: list[dict]):
    """Return list[(name, activation)]  for an observation."""
    acts, x = [], obs
    for li, ly in enumerate(conv_layers):
        x = apply_conv_layer(x, ly)
        tag = (
            f"Layer {li + 1}: "
            f"{ly['kernel_size'][0]}×{ly['kernel_size'][1]} conv, "
            f"{ly['c_in']}→{ly['c_out']}"
        )
        acts.append((tag, x.copy()))
    return acts


def get_q_values(obs_np, conv_layers, params):
    """Compute Q-values: conv → flatten → MLP → Q-head."""
    x = obs_np
    for ly in conv_layers:
        x = apply_conv_layer(x, ly)
    h = jnp.array(x.flatten())

    phi_p = params["phi"]
    linear_mods = [
        (p, phi_p[p])
        for p in sorted(phi_p)
        if hasattr(phi_p[p], "keys")
        and "w" in phi_p[p]
        and to_numpy(phi_p[p]["w"]).ndim == 2
    ]
    ln_idx = 2  # first two LN are for conv
    for _, mod in linear_mods:
        w = jnp.array(mod["w"])
        b = jnp.array(mod["b"])
        if h.shape[0] < w.shape[0]:
            h = jnp.concatenate([h, jnp.zeros(w.shape[0] - h.shape[0])])
        h = h @ w + b
        ln_key = f"phi/~/layer_norm_{ln_idx}"
        hits = [k for k in phi_p if ln_key in k]
        if hits:
            s = jnp.array(phi_p[hits[0]]["scale"])
            o = jnp.array(phi_p[hits[0]]["offset"])
            h = (h - h.mean()) / jnp.sqrt(h.var() + 1e-5) * s + o
        ln_idx += 1
        h = jax.nn.relu(h)

    q_mod = list(params["q"].values())[0]
    qw = jnp.array(q_mod["w"])
    qb = jnp.array(q_mod["b"])
    if h.shape[0] != qw.shape[0]:
        h = h[: qw.shape[0]]
    return np.asarray(h @ qw + qb)


# ── Trajectory collection ───────────────────────────────────────────────

ACTION_SYMBOLS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
ACTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}


def collect_trajectory(
    glue_state,
    env_params,
    n_steps,
    conv_layers,
    use_policy=False,
    full_params=None,
    epsilon=0.1,
):
    from foragax.registry import make as foragax_make

    env = foragax_make(**env_params)

    key = jax.random.key(42)
    key, rk = jax.random.split(key)
    obs, env_state = env.reset(rk)

    traj = []
    cumulative_reward = 0.0
    for i in range(n_steps):
        obs_np = to_numpy(obs)
        acts = get_activations(obs_np, conv_layers)
        qv = get_q_values(obs_np, conv_layers, full_params) if full_params else None

        if use_policy and full_params is not None:
            # epsilon-greedy: explore with probability epsilon
            key, eps_key = jax.random.split(key)
            if float(jax.random.uniform(eps_key)) < epsilon:
                key, ak = jax.random.split(key)
                action = int(jax.random.randint(ak, (), 0, 4))
            else:
                action = int(np.argmax(qv))
        else:
            key, ak = jax.random.split(key)
            action = int(jax.random.randint(ak, (), 0, 4))

        key, sk = jax.random.split(key)
        obs_next, env_state, reward, done, info = env.step(
            sk, env_state, jnp.array(action)
        )
        r = float(to_numpy(reward))
        cumulative_reward += r

        traj.append(
            dict(
                obs=obs_np,
                activations=acts,
                action=action,
                reward=r,
                step=i,
                cumulative_reward=cumulative_reward,
                q_values=qv,
            )
        )
        obs = obs_next
        if i % 100 == 0:
            print(f"  step {i}/{n_steps}  cumR={cumulative_reward:+.2f}")

    print(f"  Done. cumulative reward = {cumulative_reward:+.2f}")
    return traj


# ── Rendering ────────────────────────────────────────────────────────────


def render_frame_to_array(frame: dict, dpi: int = 120) -> np.ndarray:
    """Render one frame → RGB uint8 numpy array (H, W, 3)."""
    obs = frame["obs"]
    activations = frame["activations"]
    action = frame["action"]
    reward = frame["reward"]
    step = frame["step"]
    cum_r = frame["cumulative_reward"]
    qv = frame["q_values"]
    n_layers = len(activations)

    # ── figure layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(4 + 4.5 * n_layers, 5.2), dpi=dpi, facecolor="#0f0f1a")

    gs = gridspec.GridSpec(
        2,
        1 + n_layers,
        figure=fig,
        width_ratios=[1.2] + [1.8] * n_layers,
        height_ratios=[1, 0.08],
        hspace=0.28,
        wspace=0.30,
    )

    # ── Observation panel ────────────────────────────────────────────
    ax_obs = fig.add_subplot(gs[0, 0])
    obs_disp = np.clip(obs, 0, 1)
    ax_obs.imshow(obs_disp, interpolation="nearest")
    ax_obs.set_title(
        "Observation\n(9×9 RGB)", fontsize=11, fontweight="bold", color="white", pad=6
    )
    ax_obs.axis("off")

    # info bar
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis("off")
    arrow = ACTION_SYMBOLS.get(action, str(action))
    info = f"Step {step}  {arrow}  R={reward:+.1f}  ΣR={cum_r:+.1f}"
    ax_info.text(
        0.5,
        0.7,
        info,
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        transform=ax_info.transAxes,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#222244", ec="#555577", alpha=0.9),
    )

    # Q-values
    if qv is not None:
        best = int(np.argmax(qv))
        parts = []
        for a in range(len(qv)):
            s = f"{ACTION_SYMBOLS[a]}={qv[a]:+.2f}"
            if a == best:
                s = f"[{s}]"
            parts.append(s)
        ax_info.text(
            0.5,
            -0.2,
            "Q: " + "  ".join(parts),
            ha="center",
            va="center",
            fontsize=7,
            color="#aaaacc",
            transform=ax_info.transAxes,
            family="monospace",
        )

    # ── Activation panels ────────────────────────────────────────────
    for li, (layer_name, act) in enumerate(activations):
        _, _, C_out = act.shape
        n_cols = 4
        n_rows = (C_out + n_cols - 1) // n_cols

        inner = gs[0, 1 + li].subgridspec(
            n_rows + 1,
            n_cols,
            height_ratios=[0.18] + [1.0] * n_rows,
            hspace=0.08,
            wspace=0.08,
        )

        # title
        t_ax = fig.add_subplot(inner[0, :])
        t_ax.axis("off")
        t_ax.text(
            0.5,
            0.2,
            layer_name,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            transform=t_ax.transAxes,
        )

        vmax = float(np.max(np.abs(act))) or 1.0

        for ch in range(C_out):
            r, c = divmod(ch, n_cols)
            ax = fig.add_subplot(inner[r + 1, c])
            ax.imshow(
                act[:, :, ch],
                cmap="inferno",
                interpolation="nearest",
                vmin=0,
                vmax=vmax,
            )
            ax.text(
                0.06,
                0.90,
                str(ch),
                fontsize=6,
                color="white",
                transform=ax.transAxes,
                fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.5),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        for ch in range(C_out, n_rows * n_cols):
            r, c = divmod(ch, n_cols)
            ax = fig.add_subplot(inner[r + 1, c])
            ax.axis("off")

        # range label
        cb_ax = fig.add_subplot(gs[1, 1 + li])
        cb_ax.axis("off")
        cb_ax.text(
            0.5,
            0.7,
            f"activation range  0 – {vmax:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="#aaaacc",
            transform=cb_ax.transAxes,
            family="monospace",
        )

    # ── Rasterise ────────────────────────────────────────────────────
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)

    # even dimensions for h264
    h, w = rgb.shape[:2]
    if h % 2:
        rgb = rgb[: h - 1]
    if w % 2:
        rgb = rgb[:, : w - 1]
    return rgb


def create_video(trajectory, output_path, fps=10, dpi=120):
    """Write trajectory frames to MP4 via imageio."""
    import imageio.v2 as imageio

    n = len(trajectory)
    print(f"\nRendering {n} frames → {output_path}  ({fps} fps) ...")

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
    )

    for i, frame in enumerate(trajectory):
        rgb = render_frame_to_array(frame, dpi=dpi)
        writer.append_data(rgb)
        if i % 50 == 0:
            print(f"  frame {i}/{n}")

    writer.close()
    print(f"  Saved {output_path}  ({n / fps:.1f}s, {n} frames)")


def save_sample_frames(trajectory, output_path, n_samples=6, dpi=150):
    """Save a few sample frames as PNGs."""
    idxs = np.linspace(0, len(trajectory) - 1, n_samples, dtype=int)
    base, ext = os.path.splitext(output_path)
    for idx in idxs:
        frame = trajectory[idx]
        rgb = render_frame_to_array(frame, dpi=dpi)
        from PIL import Image

        p = f"{base}_frame{frame['step']:05d}.png"
        Image.fromarray(rgb).save(p)
        print(f"  {p}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CNN activations alongside observations (video)"
    )
    parser.add_argument("checkpoint", help="Path to glue_state.pkl.xz")
    parser.add_argument(
        "-o",
        "--output",
        default="cnn_activations.mp4",
        help="Output video path (default: cnn_activations.mp4)",
    )
    parser.add_argument(
        "-n",
        "--steps",
        type=int,
        default=300,
        help="Env steps to record (default: 300)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Video fps (default: 10)")
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Use learned epsilon-greedy policy (else random)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for epsilon-greedy (default: 0.1)",
    )
    parser.add_argument(
        "--samples", type=int, default=6, help="Sample PNG frames to save (default: 6)"
    )
    parser.add_argument("--env-id", default="ForagaxSquareWaveTwoBiome-v10")
    parser.add_argument("--aperture-size", type=int, default=9)
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    gs = load_glue_state(args.checkpoint)
    params = gs.agent_state.params

    print("\nParam structure:")
    for tn in sorted(params.keys()):
        tr = params[tn]
        if not hasattr(tr, "keys"):
            continue
        for mp in sorted(tr.keys()):
            mod = tr[mp]
            if not hasattr(mod, "keys"):
                continue
            for pn in sorted(mod.keys()):
                print(f"  {tn}/{mp}/{pn}: {to_numpy(mod[pn]).shape}")

    conv_layers = extract_conv_params(params)
    print(f"\n{len(conv_layers)} conv layer(s):")
    for ly in conv_layers:
        print(
            f"  {ly['name']}: {ly['kernel_size']}, "
            f"{ly['c_in']}→{ly['c_out']}, "
            f"LN={'yes' if ly['ln_scale'] is not None else 'no'}"
        )

    env_params = dict(
        env_id=args.env_id, aperture_size=args.aperture_size, observation_type="rgb"
    )
    print(f"\nEnv: {env_params}")
    policy_desc = f"learned eps-greedy (ε={args.epsilon})" if args.policy else "random"
    print(f"Collecting {args.steps} steps  (policy={policy_desc})  ...")
    traj = collect_trajectory(
        gs,
        env_params,
        args.steps,
        conv_layers,
        use_policy=args.policy,
        full_params=params,
        epsilon=args.epsilon,
    )

    print("\nSample frames:")
    save_sample_frames(traj, args.output, n_samples=args.samples)

    create_video(traj, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
