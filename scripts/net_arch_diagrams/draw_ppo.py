import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_transformer_style():
    fig, ax = plt.subplots(figsize=(10, 14))
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

    # Calculate box bounds manually because bbox padding is tricky
    def pt(ax, ay, bx, by):
        return (ax, ay), (bx, by)

    # Group backgrounds
    ax.add_patch(
        patches.FancyBboxPatch(
            (x_act - 1.5, 2.5),
            3.0,
            6.0,
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
            6.0,
            boxstyle="round,pad=0.2",
            linewidth=2,
            edgecolor="black",
            facecolor=c_bg,
            zorder=1,
        )
    )

    # Input embedding equivalents
    obs_t, obs_b, _, _ = add_box(
        x_mid, 1.0, box_w, box_h, "Flattened Obs\n(243)", c_pink
    )
    ax.text(x_mid, 0.2, "Inputs\n(9x9x3)", ha="center", va="top", fontsize=14)
    draw_arrow((x_mid, 0.2), obs_b)

    # Actor Pathway
    a_d1_t, a_d1_b, _, _ = add_box(
        x_act, 3.2, box_w, box_h, "Dense (59)\n+ tanh", c_blue
    )
    ac_c_t, ac_c_b, ac_c_l, ac_c_r = draw_concat_circle(x_act, 4.6)
    a_d2_t, a_d2_b, _, _ = add_box(
        x_act, 6.0, box_w, box_h, "Dense (192)\n+ tanh", c_blue
    )
    a_d3_t, a_d3_b, _, _ = add_box(
        x_act, 7.5, box_w, box_h, "Dense (64)\n+ tanh", c_blue
    )
    a_mean_t, a_mean_b, _, _ = add_box(x_act, 9.5, box_w, box_h, "Linear (4)", c_purple)
    a_dist_t, a_dist_b, _, _ = add_box(
        x_act, 10.8, box_w, box_h, "Categorical Distribution", c_green
    )
    ax.text(x_act, 12.0, "Output\nProbabilities", ha="center", va="bottom", fontsize=14)

    # Critic Pathway
    c_d1_t, c_d1_b, _, _ = add_box(
        x_crit, 3.2, box_w, box_h, "Dense (59)\n+ tanh", c_blue
    )
    cr_c_t, cr_c_b, cr_c_l, cr_c_r = draw_concat_circle(x_crit, 4.6)
    c_d2_t, c_d2_b, _, _ = add_box(
        x_crit, 6.0, box_w, box_h, "Dense (192)\n+ tanh", c_blue
    )
    c_d3_t, c_d3_b, _, _ = add_box(
        x_crit, 7.5, box_w, box_h, "Dense (64)\n+ tanh", c_blue
    )
    c_val_t, c_val_b, _, _ = add_box(x_crit, 9.5, box_w, box_h, "Linear (1)", c_purple)
    ax.text(x_crit, 12.0, "Value\nEstimate", ha="center", va="bottom", fontsize=14)

    # Context (Positional Encoding equivalent)
    ctx_t, ctx_b, ctx_l, ctx_r = add_box(
        x_mid, 4.6, 1.8, box_h, "Action (4)\nReward (1)", c_pink
    )

    # Text for Context branch
    ax.text(x_mid, 3.8, "Context", ha="center", va="center", fontsize=14)
    draw_arrow((x_mid, 3.8), ctx_b)

    # Draw connections
    # Obs to Denses
    ax.plot([x_mid, x_mid], [obs_t[1], 2.0], color="black", linewidth=1.5, zorder=2)
    ax.plot([x_act, x_crit], [2.0, 2.0], color="black", linewidth=1.5, zorder=2)
    draw_arrow((x_act, 2.0), a_d1_b)
    draw_arrow((x_crit, 2.0), c_d1_b)

    # Actor forward
    draw_arrow(a_d1_t, ac_c_b)
    draw_arrow(ac_c_t, a_d2_b)
    draw_arrow(a_d2_t, a_d3_b)
    draw_arrow(a_d3_t, a_mean_b)
    draw_arrow(a_mean_t, a_dist_b)
    draw_arrow(a_dist_t, (x_act, 12.0))

    # Critic forward
    draw_arrow(c_d1_t, cr_c_b)
    draw_arrow(cr_c_t, c_d2_b)
    draw_arrow(c_d2_t, c_d3_b)
    draw_arrow(c_d3_t, c_val_b)
    draw_arrow(c_val_t, (x_crit, 12.0))

    # Context to Concat circles
    draw_arrow(ctx_l, ac_c_r)
    draw_arrow(ctx_r, cr_c_l)

    plt.xlim(0, 10)
    plt.ylim(-0.5, 13)

    # Figure 1: ... text at bottom
    ax.text(
        5.0,
        -1.0,
        "Figure 1: The ActorCriticMLP - model architecture.",
        ha="center",
        va="center",
        fontsize=16,
        family="serif",
    )

    plt.savefig("actor_critic_transformer_arch.png", bbox_inches="tight", dpi=300)
    print("Saved as 'actor_critic_transformer_arch.png'")


if __name__ == "__main__":
    draw_transformer_style()
