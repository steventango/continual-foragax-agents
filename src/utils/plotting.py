import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)


def plot(ax, data, label=None):
    mean, ste, runs = data
    (base,) = ax.plot(mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    ax.fill_between(
        range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4
    )


def _calculate_cost(line, x, y, angle, current_offset, placed_bboxes, **kwargs):
    """Helper to calculate the cost of a potential label position."""
    ax = line.axes
    label = line.get_label()
    fontsize = kwargs.get("fontsize", plt.rcParams["font.size"])

    # 2. Calculate final label position and its bounding box
    perp_angle_rad = np.deg2rad(angle + 90)
    dx_offset = current_offset * np.cos(perp_angle_rad)
    dy_offset = current_offset * np.sin(perp_angle_rad)

    pos_disp = ax.transData.transform((x, y))
    label_center_disp = (pos_disp[0] + dx_offset, pos_disp[1] + dy_offset)

    width_pt = len(label) * fontsize * 0.6
    height_pt = fontsize

    candidate_bbox = Bbox.from_bounds(
        label_center_disp[0] - width_pt / 2,
        label_center_disp[1] - height_pt / 2,
        width_pt,
        height_pt,
    )

    # --- Calculate the cost for this position ---
    # 1. Overlap with other LABELS
    label_overlap_cost = 0
    if any(candidate_bbox.overlaps(bbox) for bbox in placed_bboxes):
        label_overlap_cost = 10000  # Very high cost for text overlap

    # 2. Overlap with other LINES and CIs (now additive)
    line_overlap_cost = 0
    for other_line in ax.get_lines():
        if (
            other_line is line
            or not other_line.get_label()
            or other_line.get_label().startswith("_")
        ):
            continue

        other_x, other_y = other_line.get_data()
        other_mask = np.isfinite(other_y)
        if not np.any(other_mask):
            continue

        other_x, other_y = other_x[other_mask], other_y[other_mask]
        other_pts_disp = ax.transData.transform(np.vstack([other_x, other_y]).T)

        for j in range(len(other_pts_disp) - 1):
            p1 = other_pts_disp[j]
            p2 = other_pts_disp[j + 1]
            segment_bbox = Bbox([p1, p2])
            if candidate_bbox.overlaps(segment_bbox):
                line_overlap_cost += 500  # Add cost for each intersection
                break  # Break to avoid counting multiple segments of the same line

    ci_overlap_cost = 0
    for collection in ax.collections:
        if not hasattr(collection, "get_paths"):
            continue
        for path in collection.get_paths():
            path_transformed = path.transformed(ax.transData)
            if path_transformed.intersects_bbox(candidate_bbox):
                ci_overlap_cost += 500  # Add cost for each CI intersection
                break  # Break to avoid counting the same CI multiple times

    # 3. Slope cost (prefer flatter parts of the line)
    slope_cost = abs(angle) * 0.5

    # 4. Y-Clipping cost (avoid placing labels near the edge)
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    y_rel = (y - ylim[0]) / y_range
    clipping_cost = 0
    if y_rel < 0.1 or y_rel > 0.9:
        clipping_cost = 100

    # 5. Axes clipping cost (avoid placing labels outside the axes)
    ax_bbox_disp = ax.get_window_extent()
    if not (
        ax_bbox_disp.contains(candidate_bbox.x0, candidate_bbox.y0)
        and ax_bbox_disp.contains(candidate_bbox.x1, candidate_bbox.y1)
    ):
        clipping_cost += 1000

    # 6. Right-side clipping cost
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    x_rel = (x - xlim[0]) / x_range
    if x_rel > 0.9:
        clipping_cost += (x_rel - 0.9) * 1000

    total_cost = (
        label_overlap_cost
        + line_overlap_cost
        + ci_overlap_cost
        + slope_cost
        + clipping_cost
    )

    return total_cost, candidate_bbox


def label_lines(ax, align=False, xvals=None, offset_range=(6, 24), **kwargs):
    """
    Adds labels to a list of Matplotlib line plots, attempting to position them
    to minimize overlap with other labels and lines.

    Args:
        ax (matplotlib.axes.Axes): The axes object containing the lines to label.
        align (bool, optional): If True, the labels will be rotated to match the line's angle.
        xvals (list of float, optional): A list of x-coordinates for each line. If not provided,
                                         the function will automatically find good positions.
        offset_range (tuple of float, optional): A tuple (min, max) defining the range of
                                                  perpendicular distances (in points) to search for
                                                  the best label position.
        **kwargs: Additional keyword arguments to pass to the annotation.
    """
    placed_bboxes = []  # List to store bounding boxes of labels already placed.
    lines = ax.get_lines()

    # If xvals are provided, use them.
    if xvals:
        # In manual mode, we can't easily decide the best offset, so we use the average.
        offset = np.mean(offset_range)
        for line, x in zip(lines, xvals, strict=True):
            label_line(line, x, align=align, offset=offset, **kwargs)
        return

    # Automatic placement
    for line in lines:
        label = line.get_label()
        if not label or label.startswith("_"):
            continue

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        # Get the visible x-range of the plot
        xlim = ax.get_xlim()
        visible_mask = (xdata >= xlim[0]) & (xdata <= xlim[1]) & np.isfinite(ydata)
        visible_xdata = xdata[visible_mask]
        visible_ydata = ydata[visible_mask]

        if len(visible_xdata) < 2:
            continue

        # --- Find the best position for the label ---
        best_pos = None
        best_offset = None
        min_cost = float("inf")

        # Sample a number of points along the line
        candidate_indices = np.linspace(0, len(visible_xdata) - 1, num=30, dtype=int)[
            1:-1
        ]

        # Sample a range of offsets to find the best one
        offset_candidates = np.linspace(
            offset_range[0], offset_range[1], num=5
        )  # e.g., [6, 12, 18, 24]

        for i in candidate_indices:
            # Ensure we have a segment to calculate the angle
            if i == 0:
                continue

            x1, y1 = visible_xdata[i - 1], visible_ydata[i - 1]
            x2, y2 = visible_xdata[i], visible_ydata[i]
            x, y = x2, y2

            # --- Calculate properties of the candidate position ---
            p1_disp, p2_disp = ax.transData.transform([(x1, y1), (x2, y2)])
            angle = np.rad2deg(
                np.arctan2(p2_disp[1] - p1_disp[1], p2_disp[0] - p1_disp[0])
            )
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180

            # --- Try different offsets for this point ---
            for offset_val in offset_candidates:
                # --- Calculate cost for both above and below positions ---
                cost_above, bbox_above = _calculate_cost(
                    line, x, y, angle, offset_val, placed_bboxes, **kwargs
                )
                cost_below, bbox_below = _calculate_cost(
                    line, x, y, angle, -offset_val, placed_bboxes, **kwargs
                )

                # Choose the better of the two options
                current_cost, current_bbox, current_offset = (
                    (cost_above, bbox_above, offset_val)
                    if cost_above <= cost_below
                    else (cost_below, bbox_below, -offset_val)
                )

                if current_cost < min_cost:
                    min_cost = current_cost
                    best_pos = x
                    best_offset = current_offset

        if best_pos and best_offset is not None:
            # Place the label at the best found position with the best offset
            label_line(line, best_pos, align=align, offset=best_offset, **kwargs)

            # Add the new label's bounding box to the list for future checks
            # We need to recalculate the bbox for the final chosen position
            final_bbox_info = _get_label_info(
                line, best_pos, align, best_offset, **kwargs
            )
            placed_bboxes.append(final_bbox_info["bbox"])


def _get_label_info(line, x, align, offset, **kwargs):
    """Helper function to calculate label position and bbox without drawing."""
    ax = line.axes
    xdata, ydata = line.get_xdata(), line.get_ydata()

    # Interpolate y
    index = np.searchsorted(xdata, x)
    if index == 0:
        index = 1
    if index == len(xdata):
        index = len(xdata) - 1
    x1, y1 = xdata[index - 1], ydata[index - 1]
    x2, y2 = xdata[index], ydata[index]
    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    # Calculate angle
    p1_disp, p2_disp = ax.transData.transform([(x1, y1), (x2, y2)])
    angle = np.rad2deg(np.arctan2(p2_disp[1] - p1_disp[1], p2_disp[0] - p1_disp[0]))
    if align:
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
    else:
        angle = 0

    # Calculate final position and bbox
    perp_angle_rad = np.deg2rad(angle + 90)
    dx_offset = offset * np.cos(perp_angle_rad)
    dy_offset = offset * np.sin(perp_angle_rad)
    pos_disp = ax.transData.transform((x, y))
    label_center_disp = (pos_disp[0] + dx_offset, pos_disp[1] + dy_offset)

    label = kwargs.get("label", line.get_label())
    fontsize = kwargs.get("fontsize", plt.rcParams["font.size"])
    width_pt = len(label) * fontsize * 0.6
    height_pt = fontsize

    bbox = Bbox.from_bounds(
        label_center_disp[0] - width_pt / 2,
        label_center_disp[1] - height_pt / 2,
        width_pt,
        height_pt,
    )

    return {"x": x, "y": y, "angle": angle, "bbox": bbox}


def label_line(line, x=None, label=None, align=False, offset=12, **kwargs):
    """
    Adds a label to a Matplotlib line plot, positioned close to the line.
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if x is None:
        xlim = ax.get_xlim()
        visible_mask = (xdata >= xlim[0]) & (xdata <= xlim[1]) & np.isfinite(ydata)
        visible_xdata = xdata[visible_mask]
        if len(visible_xdata) == 0:
            return
        x = np.median(visible_xdata)

    index = np.searchsorted(xdata, x)
    if index == 0:
        index = 1
    if index == len(xdata):
        index = len(xdata) - 1

    x1, y1 = xdata[index - 1], ydata[index - 1]
    x2, y2 = xdata[index], ydata[index]

    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    if label is None:
        label = line.get_label()

    if align:
        p1_disp, p2_disp = ax.transData.transform([(x1, y1), (x2, y2)])
        angle = np.rad2deg(np.arctan2(p2_disp[1] - p1_disp[1], p2_disp[0] - p1_disp[0]))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
    else:
        angle = 0

    perp_angle_rad = np.deg2rad(angle + 90)
    dx_offset = offset * np.cos(perp_angle_rad)
    dy_offset = offset * np.sin(perp_angle_rad)

    text_kwargs = {
        "rotation": angle,
        "ha": "center",
        "va": "center",
        "color": line.get_color(),
        "xytext": (dx_offset, dy_offset),
        "textcoords": "offset points",
    }
    text_kwargs.update(kwargs)

    ax.annotate(label, xy=(x, y), **text_kwargs)
