import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_transformer_rtu_style():
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.axis("off")

    c_pink = "#ffcdd2"
    c_blue = "#b3e5fc"
    c_yellow = "#ffecb3"
    c_purple = "#e1bee7"
    c_green = "#c8e6c9"
    c_bg = "#f5f5f5"

    def add_box(x, y, w, h, text, color, style="round,pad=0.2", lw=2):
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=style,
            linewidth=lw,
            edgecolor="black",
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
            family="sans-serif",
        )
        return (
            (x, y + h / 2 + 0.2),
            (x, y - h / 2 - 0.2),
            (x - w / 2 - 0.2, y),
            (x + w / 2 + 0.2, y),
        )

    def draw_arrow(pt1, pt2, style="-|>", rad=0, lw=1.5):
        arrow = patches.FancyArrowPatch(
            pt1,
            pt2,
            connectionstyle=f"arc3,rad={rad}",
            color="black",
            arrowstyle=style,
            mutation_scale=15,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(arrow)

    def draw_concat_circle(x, y, radius=0.25):
        circle = patches.Circle(
            (x, y), radius, linewidth=2, edgecolor="black", facecolor="white", zorder=3
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            "C",
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
            fontfamily="sans-serif",
            fontweight="bold",
        )
        return (x, y + radius), (x, y - radius), (x - radius, y), (x + radius, y)

    x_act = 2.5
    x_mid = 5.0
    x_crit = 7.5
    box_w = 2.4
    box_h = 0.7

    # Group backgrounds
    ax.add_patch(
        patches.FancyBboxPatch(
            (x_act - 1.5, 2.5),
            3.0,
            7.5,
            boxstyle="round,pad=0.2",
            linewidth=2,
            edgecolor="black",
            facecolor=c_bg,
            zorder=1,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x_crit - 1.5, 2.5),
            3.0,
            7.5,
            boxstyle="round,pad=0.2",
            linewidth=2,
            edgecolor="black",
            facecolor=c_bg,
            zorder=1,
        )
    )

    # Inputs
    obs_t, obs_b, _, _ = add_box(
        x_mid, 1.0, box_w, box_h, "Flattened Obs\n(243)", c_pink
    )
    ax.text(x_mid, 0.2, "Inputs\n(9x9x3)", ha="center", va="top", fontsize=14)
    draw_arrow((x_mid, 0.2), obs_b)

    # Actor Pathway
    a_d1_t, a_d1_b, _, _ = add_box(
        x_act, 3.2, box_w, box_h, "Dense (59)\n+ tanh", c_blue
    )
    ac_c_t, ac_c_b, ac_c_l, ac_c_r = draw_concat_circle(x_act, 4.3)

    # RTU Layer
    a_rtu_t, a_rtu_b, a_rtu_l, a_rtu_r = add_box(
        x_act, 5.7, box_w, box_h, "RTU Layer (192)", c_yellow
    )

    # Hidden State Loop (Recurrence) for RTU
    ax.plot(
        [x_act + 1.2, x_act + 1.5], [5.7, 5.7], color="black", linewidth=1.5, zorder=2
    )
    ax.plot(
        [x_act + 1.5, x_act + 1.5], [5.7, 6.3], color="black", linewidth=1.5, zorder=2
    )
    ax.plot(
        [x_act + 1.5, x_act - 1.4], [6.3, 6.3], color="black", linewidth=1.5, zorder=2
    )
    ax.plot(
        [x_act - 1.4, x_act - 1.4], [6.3, 5.7], color="black", linewidth=1.5, zorder=2
    )
    draw_arrow((x_act - 1.4, 5.7), a_rtu_l)

    # Skip Concat
    a_sc_t, a_sc_b, a_sc_l, a_sc_r = draw_concat_circle(x_act, 7.1)

    a_d2_t, a_d2_b, _, _ = add_box(
        x_act, 8.5, box_w, box_h, "Dense (64)\n+ tanh", c_blue
    )
    a_mean_t, a_mean_b, _, _ = add_box(
        x_act, 10.5, box_w, box_h, "Linear (4)", c_purple
    )
    a_dist_t, a_dist_b, _, _ = add_box(
        x_act, 11.8, box_w, box_h, "Categorical Distribution", c_green
    )
    ax.text(x_act, 13.0, "Output\nProbabilities", ha="center", va="bottom", fontsize=14)

    # Critic Pathway
    c_d1_t, c_d1_b, _, _ = add_box(
        x_crit, 3.2, box_w, box_h, "Dense (59)\n+ tanh", c_blue
    )
    cr_c_t, cr_c_b, cr_c_l, cr_c_r = draw_concat_circle(x_crit, 4.3)

    # RTU Layer
    c_rtu_t, c_rtu_b, c_rtu_l, c_rtu_r = add_box(
        x_crit, 5.7, box_w, box_h, "RTU Layer (192)", c_yellow
    )

    # Hidden State Loop (Recurrence) for RTU
    ax.plot(
        [x_crit + 1.2, x_crit + 1.5], [5.7, 5.7], color="black", linewidth=1.5, zorder=2
    )
    ax.plot(
        [x_crit + 1.5, x_crit + 1.5], [5.7, 6.3], color="black", linewidth=1.5, zorder=2
    )
    ax.plot(
        [x_crit + 1.5, x_crit - 1.4], [6.3, 6.3], color="black", linewidth=1.5, zorder=2
    )
    ax.plot(
        [x_crit - 1.4, x_crit - 1.4], [6.3, 5.7], color="black", linewidth=1.5, zorder=2
    )
    draw_arrow((x_crit - 1.4, 5.7), c_rtu_l)

    # Skip Concat
    c_sc_t, c_sc_b, c_sc_l, c_sc_r = draw_concat_circle(x_crit, 7.1)

    c_d2_t, c_d2_b, _, _ = add_box(
        x_crit, 8.5, box_w, box_h, "Dense (64)\n+ tanh", c_blue
    )
    c_val_t, c_val_b, _, _ = add_box(x_crit, 10.5, box_w, box_h, "Linear (1)", c_purple)
    ax.text(x_crit, 13.0, "Value\nEstimate", ha="center", va="bottom", fontsize=14)

    # Context (Positional Encoding equivalent)
    ctx_text = "Action (4)\nReward (1)"
    ctx_t, ctx_b, ctx_l, ctx_r = add_box(x_mid, 4.3, 1.8, box_h, ctx_text, c_pink)
    ax.text(x_mid, 3.5, "Context", ha="center", va="center", fontsize=14)
    draw_arrow((x_mid, 3.5), ctx_b)

    # Connect inputs to denses
    ax.plot([x_mid, x_mid], [obs_t[1], 2.0], color="black", linewidth=1.5, zorder=2)
    ax.plot([x_act, x_crit], [2.0, 2.0], color="black", linewidth=1.5, zorder=2)
    draw_arrow((x_act, 2.0), a_d1_b)
    draw_arrow((x_crit, 2.0), c_d1_b)

    # ---------------- ACTOR ARROWS ----------------
    draw_arrow(a_d1_t, ac_c_b)
    draw_arrow(ctx_l, ac_c_r)  # Context to first concat

    # First concat to RTU AND Skip Connection
    ax.plot([x_act, x_act], [ac_c_t[1], 4.8], color="black", linewidth=1.5, zorder=2)
    ax.plot([x_act, x_act - 1.6], [4.8, 4.8], color="black", linewidth=1.5, zorder=2)
    ax.plot(
        [x_act - 1.6, x_act - 1.6], [4.8, 7.1], color="black", linewidth=1.5, zorder=2
    )
    draw_arrow((x_act - 1.6, 7.1), (x_act - 0.25, 7.1))  # Skip to Concat

    draw_arrow((x_act, 4.8), a_rtu_b)  # Concat to RTU

    draw_arrow(a_rtu_t, a_sc_b)  # RTU to Skip Concat (new embedding 192)
    draw_arrow(a_sc_t, a_d2_b)  # Skip Concat to Dense2 (features: 256)
    draw_arrow(a_d2_t, a_mean_b)
    draw_arrow(a_mean_t, a_dist_b)
    draw_arrow(a_dist_t, (x_act, 13.0))

    # ---------------- CRITIC ARROWS ----------------
    draw_arrow(c_d1_t, cr_c_b)
    draw_arrow(ctx_r, cr_c_l)  # Context to first concat

    # First concat to RTU AND Skip Connection
    ax.plot([x_crit, x_crit], [cr_c_t[1], 4.8], color="black", linewidth=1.5, zorder=2)
    ax.plot([x_crit, x_crit - 1.6], [4.8, 4.8], color="black", linewidth=1.5, zorder=2)
    ax.plot(
        [x_crit - 1.6, x_crit - 1.6], [4.8, 7.1], color="black", linewidth=1.5, zorder=2
    )
    draw_arrow((x_crit - 1.6, 7.1), (x_crit - 0.25, 7.1))  # Skip to Concat

    draw_arrow((x_crit, 4.8), c_rtu_b)  # Concat to RTU

    draw_arrow(c_rtu_t, c_sc_b)  # RTU to Skip Concat (new embedding 192)
    draw_arrow(c_sc_t, c_d2_b)  # Skip Concat to Dense2 (features: 256)
    draw_arrow(c_d2_t, c_val_b)
    draw_arrow(c_val_t, (x_crit, 13.0))

    plt.xlim(-0.2, 10.2)
    plt.ylim(-0.5, 14.0)

    # Figure text
    ax.text(
        5.0,
        -1.0,
        "Figure 1: The RealTimeActorCriticMLP (PPO-RTU) model architecture.",
        ha="center",
        va="center",
        fontsize=16,
        family="serif",
    )

    plt.savefig("ppo_rtu_transformer_arch.png", bbox_inches="tight", dpi=300)
    print("Saved as 'ppo_rtu_transformer_arch.png'")


if __name__ == "__main__":
    draw_transformer_rtu_style()
