from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .helpersTracking import combined_summary_table, gt_lookup, particle_type


DEFAULT_METHOD_ORDER = ["Regular pipeline", "Separated pipeline"]
DEFAULT_METHOD_COLORS = {
    "Regular pipeline": "#4c78a8",
    "Separated pipeline": "#f58518",
}
DEFAULT_TYPE_COLORS = {
    "ligand": "#1f9e89",
    "receptor": "#d1495b",
    "unassigned": "#6b7280",
}


def _with_defaults(values, defaults):
    merged = dict(defaults)
    if values:
        merged.update(values)
    return merged


def _metrics_by_method(method_metrics):
    if isinstance(method_metrics, dict):
        return dict(method_metrics)
    return {metrics["method"]: metrics for metrics in method_metrics}


def _resolve_method_order(method_order, *method_mappings):
    available = []
    for mapping in method_mappings:
        available.extend(list(mapping.keys()))
    available = list(dict.fromkeys(available))

    if method_order is None:
        preferred = [method for method in DEFAULT_METHOD_ORDER if method in available]
        rest = [method for method in available if method not in preferred]
        return preferred + rest

    return [method for method in method_order if method in available]


def build_state_summary(method_metrics):
    metrics_by_method = _metrics_by_method(method_metrics)
    state_tables = [
        metrics["states"]
        for metrics in metrics_by_method.values()
        if len(metrics.get("states", []))
    ]

    if not state_tables:
        return pd.DataFrame(
            columns=["method", "type", "state", "n_frames", "recovered_frames", "coverage_pct"]
        )

    state_summary = (
        pd.concat(state_tables, ignore_index=True)
        .groupby(["method", "type", "state"], as_index=False)
        .agg(
            n_frames=("n_frames", "sum"),
            recovered_frames=("recovered_frames", "sum"),
        )
    )
    state_summary["coverage_pct"] = (
        100 * state_summary["recovered_frames"] / state_summary["n_frames"]
    )
    return state_summary


def plot_ligand_receptor_frame(
    sim,
    frame_id=0,
    show_ids=False,
    type_colors=None,
    show=True,
):
    type_colors = _with_defaults(type_colors, DEFAULT_TYPE_COLORS)
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = np.percentile(sim.frames, 99.8)
    ax.imshow(sim.frames[frame_id], cmap="gray", vmin=0, vmax=vmax)

    for traj in sim.trajectories_GT:
        pos = traj.get_position_at_frame(frame_id)
        if pos is None:
            continue

        row, col = pos
        gt_type = particle_type(traj)
        is_receptor = gt_type == "receptor"
        is_bound = traj.get_state_at_frame(frame_id) == "bound"

        ax.scatter(
            col,
            row,
            s=75 if is_receptor else 26,
            c=type_colors.get(gt_type, "white"),
            marker="s" if is_receptor else "o",
            edgecolors="yellow" if is_bound else "black",
            linewidths=1.2 if is_bound else 0.4,
            alpha=0.85,
        )

        if show_ids:
            ax.text(col + 1, row + 1, str(traj.id), color="white", fontsize=8)

    ax.set_title(f"Ligand-receptor frame {frame_id}")
    ax.axis("off")

    if show:
        plt.show()

    return fig


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_gt_frame(ax, sim, frame_id, type_colors=None):
    type_colors = _with_defaults(type_colors, DEFAULT_TYPE_COLORS)
    vmax = np.percentile(sim.frames, 99.8)
    ax.imshow(sim.frames[frame_id], cmap="gray", vmin=0, vmax=vmax)

    for traj in sim.trajectories_GT:
        pos = traj.get_position_at_frame(frame_id)
        if pos is None:
            continue

        row, col = pos
        gt_type = particle_type(traj)
        is_bound = traj.get_state_at_frame(frame_id) == "bound"
        ax.scatter(
            col,
            row,
            s=70 if gt_type == "receptor" else 28,
            c=type_colors.get(gt_type, "white"),
            marker="s" if gt_type == "receptor" else "o",
            edgecolors="yellow" if is_bound else "black",
            linewidths=1.2 if is_bound else 0.4,
            alpha=0.9,
        )

    ax.set_title(f"GT frame {frame_id}: ligands, receptors, bound states", fontsize=10)
    ax.axis("off")


def plot_tracks_on_axis(
    ax,
    frames,
    trajectories,
    trajectories_GT,
    frame_id,
    title,
    type_colors=None,
):
    type_colors = _with_defaults(type_colors, DEFAULT_TYPE_COLORS)
    gt_by_id = gt_lookup(trajectories_GT)
    vmax = np.percentile(frames, 99.8)
    ax.imshow(frames[frame_id], cmap="gray", vmin=0, vmax=vmax)

    for traj in trajectories:
        if frame_id < traj.start_frame or traj.length() < 2:
            continue

        valid = [
            pos for frame, pos in zip(traj.frames(), traj.positions)
            if frame <= frame_id
        ]
        if len(valid) < 2:
            continue

        positions = np.asarray(valid)
        gt_traj = gt_by_id.get(traj.id)
        color = (
            type_colors.get(particle_type(gt_traj), type_colors["unassigned"])
            if gt_traj is not None
            else type_colors["unassigned"]
        )
        ax.plot(positions[:, 1], positions[:, 0], color=color, linewidth=1.7, alpha=0.9)
        ax.plot(positions[-1, 1], positions[-1, 0], marker="+", color=color, markersize=6)

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_coverage_panel(
    ax,
    summary_table,
    method_order=None,
    method_colors=None,
    populations=("ligand", "receptor"),
    tolerance_px=None,
):
    method_colors = _with_defaults(method_colors, DEFAULT_METHOD_COLORS)
    method_order = _resolve_method_order(
        method_order,
        {method: None for method in summary_table["method"].dropna().unique()},
    )

    width = 0.7 / max(len(method_order), 1)
    x = np.arange(len(populations))

    for idx, method in enumerate(method_order):
        values = []
        for pop in populations:
            row = summary_table.loc[
                (summary_table["method"] == method)
                & (summary_table["population"] == pop),
                "mean_coverage_pct",
            ]
            values.append(float(row.iloc[0]) if len(row) else 0.0)

        offset = (idx - (len(method_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=method_colors.get(method, "#6b7280"),
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([pop.capitalize() for pop in populations])
    ylabel = "GT frame coverage (%)"
    if tolerance_px is not None:
        ylabel = f"GT frame coverage within {tolerance_px:g} px (%)"
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    ax.set_title("Coverage by particle type", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def plot_error_panel(
    ax,
    method_metrics,
    method_order=None,
    method_colors=None,
    populations=("ligand", "receptor"),
):
    metrics_by_method = _metrics_by_method(method_metrics)
    method_order = _resolve_method_order(method_order, metrics_by_method)
    method_colors = _with_defaults(method_colors, DEFAULT_METHOD_COLORS)

    data = []
    colors = []
    positions = []
    tick_positions = []

    width = 0.7 / max(len(method_order), 1)
    for pop_idx, pop in enumerate(populations):
        base_position = pop_idx + 1.0
        tick_positions.append(base_position)
        for method_idx, method in enumerate(method_order):
            metrics = metrics_by_method[method]
            errors = metrics["errors"]
            values = errors.loc[errors["type"] == pop, "error_px"].dropna().to_numpy()
            data.append(values if len(values) else np.asarray([np.nan]))
            colors.append(method_colors.get(method, "#6b7280"))
            positions.append(
                base_position + (method_idx - (len(method_order) - 1) / 2) * width
            )

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pop.capitalize() for pop in populations])
    ax.set_ylabel("Position error (px)")
    ax.set_title("Localization error distribution", fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def plot_state_panel(
    ax,
    state_summary,
    method_order=None,
    method_colors=None,
    particle_type_name="ligand",
    states_order=("free", "bound"),
    tolerance_px=None,
):
    method_colors = _with_defaults(method_colors, DEFAULT_METHOD_COLORS)
    ligand_state = state_summary[state_summary["type"] == particle_type_name].copy()
    method_order = _resolve_method_order(
        method_order,
        {method: None for method in ligand_state["method"].dropna().unique()},
    )
    states = [state for state in states_order if state in set(ligand_state["state"])]

    if not states:
        ax.text(
            0.5,
            0.5,
            f"No {particle_type_name} state data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    width = 0.7 / max(len(method_order), 1)
    x = np.arange(len(states))
    for idx, method in enumerate(method_order):
        values = []
        for state in states:
            row = ligand_state[
                (ligand_state["method"] == method)
                & (ligand_state["state"] == state)
            ]
            values.append(float(row["coverage_pct"].iloc[0]) if len(row) else 0.0)

        offset = (idx - (len(method_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=method_colors.get(method, "#6b7280"),
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([state.capitalize() for state in states])
    ylabel = f"{particle_type_name.capitalize()} coverage (%)"
    if tolerance_px is not None:
        ylabel = f"{particle_type_name.capitalize()} coverage within {tolerance_px:g} px (%)"
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    ax.set_title(f"{particle_type_name.capitalize()} recovery by binding state", fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def plot_peak_count_panel(
    ax,
    peaks_full,
    peaks_masked,
    method_colors=None,
    full_label="All peaks",
    masked_label="After static mask",
):
    method_colors = _with_defaults(method_colors, DEFAULT_METHOD_COLORS)
    full_counts = np.asarray([len(frame_peaks) for frame_peaks in peaks_full], dtype=float)
    masked_counts = np.asarray([len(frame_peaks) for frame_peaks in peaks_masked], dtype=float)
    frames_axis = np.arange(len(full_counts))

    ax.plot(
        frames_axis,
        full_counts,
        color=method_colors.get("Regular pipeline", "#4c78a8"),
        linewidth=1.8,
        label=full_label,
    )
    ax.plot(
        frames_axis,
        masked_counts,
        color=method_colors.get("Separated pipeline", "#f58518"),
        linewidth=1.8,
        label=masked_label,
    )
    ax.fill_between(
        frames_axis,
        masked_counts,
        full_counts,
        color="#9ca3af",
        alpha=0.25,
        label="Removed",
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Localized peaks")
    ax.set_title("Mask effect on detections", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)


def plot_ligand_receptor_comparison_figure(
    sim,
    frames,
    trajectories_GT,
    method_trajectories,
    method_metrics,
    summary_table=None,
    state_summary=None,
    frame_id=None,
    save_base=None,
    method_order=None,
    method_colors=None,
    type_colors=None,
    track_titles=None,
    figure_title=None,
    tolerance_px=None,
    figsize=(15, 9),
):
    """
    Plot the six-panel ligand-receptor comparison without reading notebook globals.

    Parameters
    ----------
    method_trajectories : dict
        Mapping from method name to assigned trajectories.
    method_metrics : dict or sequence
        Mapping from method name to metrics dictionaries, or a sequence of
        metrics dictionaries returned by ``evaluate_tracking_method``.
    save_base : path-like, optional
        If provided, save PNG and PDF versions using this path stem.
    """
    method_trajectories = dict(method_trajectories)
    metrics_by_method = _metrics_by_method(method_metrics)
    method_order = _resolve_method_order(
        method_order,
        method_trajectories,
        metrics_by_method,
    )
    method_colors = _with_defaults(method_colors, DEFAULT_METHOD_COLORS)
    type_colors = _with_defaults(type_colors, DEFAULT_TYPE_COLORS)
    track_titles = track_titles or {}

    if frame_id is None:
        frame_id = min(20, len(frames) - 1)
    if summary_table is None:
        summary_table = combined_summary_table(
            *[metrics_by_method[method] for method in method_order]
        )
    if state_summary is None:
        state_summary = build_state_summary(
            {method: metrics_by_method[method] for method in method_order}
        )
    if figure_title is None:
        figure_title = " vs ".join(method_order)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    plot_gt_frame(axes[0], sim, frame_id, type_colors=type_colors)

    for axis_idx, method in enumerate(method_order[:2], start=1):
        title = track_titles.get(method, f"{method} tracking")
        plot_tracks_on_axis(
            axes[axis_idx],
            frames,
            method_trajectories[method],
            trajectories_GT,
            frame_id,
            title,
            type_colors=type_colors,
        )

    for axis_idx in range(1 + len(method_order[:2]), 3):
        axes[axis_idx].axis("off")

    plot_coverage_panel(
        axes[3],
        summary_table,
        method_order=method_order,
        method_colors=method_colors,
        tolerance_px=tolerance_px,
    )
    plot_error_panel(
        axes[4],
        metrics_by_method,
        method_order=method_order,
        method_colors=method_colors,
    )
    plot_state_panel(
        axes[5],
        state_summary,
        method_order=method_order,
        method_colors=method_colors,
        tolerance_px=tolerance_px,
    )

    for label, ax in zip("ABCDEF", axes):
        add_panel_label(ax, label)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=type_colors["ligand"],
            markeredgecolor="black",
            label="Ligand",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=type_colors["receptor"],
            markeredgecolor="black",
            label="Receptor",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="yellow",
            label="Bound state",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(figure_title, y=1.04, fontsize=14)
    fig.tight_layout()

    saved_paths = []
    if save_base is not None:
        save_base = Path(save_base)
        png_path = save_base.with_suffix(".png")
        pdf_path = save_base.with_suffix(".pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        saved_paths.extend([png_path, pdf_path])

    return fig, saved_paths

def linear_trajectories_visualizer(
    trajectories_new,
    trajectories_GT,
    quality_tolerance=3.0,
    separate_types=True,
    show_bound_states=True,
    show_unassigned=True,
    show_labels=False,
    title=None,
    figsize=None,
    save_path=None,
    return_fig=False,
):
    """
    Visualize tracking quality as temporal bars aligned to ground truth.

    Ground-truth trajectories are shown as pale horizontal bars. Estimated
    trajectories are overlaid on the assigned GT row and colored by their mean
    localization error on overlapping frames.

    Parameters
    ----------
    trajectories_new : list[Trajectory]
        Tracked trajectories after assignment to GT ids.
    trajectories_GT : list[Trajectory]
        Ground-truth trajectories.
    quality_tolerance : float
        Error in pixels below which a tracked segment is counted as good.
    separate_types : bool
        If True, create separate panels for receptors and ligands when
        ``particle_type`` metadata is available.
    show_bound_states : bool
        If True, draw a small purple underline where GT states are "bound".
    show_unassigned : bool
        If True, display unassigned tracks in a separate panel.
    show_labels : bool
        If True, annotate tracked segments with error and coverage.
    title : str or None
        Optional figure title.
    figsize : tuple or None
        Optional matplotlib figure size.
    save_path : str or None
        Optional path where the figure is saved.
    return_fig : bool
        If True, return ``(fig, axes)``. Defaults to False to keep notebook
        output clean when the function is called as the last line of a cell.

    Returns
    -------
    tuple or None
        ``(fig, axes)`` if ``return_fig=True``, otherwise ``None``.
    """
    from matplotlib.lines import Line2D

    def _particle_type(traj):
        return getattr(traj, "particle_type", "particle") or "particle"

    def _ordered_types(types):
        preferred = ["receptor", "ligand", "particle"]
        ordered = [t for t in preferred if t in types]
        ordered += sorted(t for t in types if t not in preferred)
        return ordered

    def _overlap_interval(traj_new, traj_gt):
        start = max(traj_new.start_frame, traj_gt.start_frame)
        end = min(traj_new.end_frame, traj_gt.end_frame)
        if start > end:
            return None
        return start, end

    def _overlap_distances(traj_new, traj_gt):
        interval = _overlap_interval(traj_new, traj_gt)
        if interval is None:
            return []

        start, end = interval
        distances = []
        for frame in range(start, end + 1):
            pos_new = traj_new.get_position_at_frame(frame)
            pos_gt = traj_gt.get_position_at_frame(frame)

            if pos_new is None or pos_gt is None:
                continue

            distances.append(
                float(np.linalg.norm(np.asarray(pos_new) - np.asarray(pos_gt)))
            )

        return distances

    def _quality(mean_error):
        if not np.isfinite(mean_error):
            return "#6b7280", "no overlap"
        if mean_error <= quality_tolerance:
            return "#2ca25f", f"err <= {quality_tolerance:g} px"
        if mean_error <= 2 * quality_tolerance:
            return "#f59e0b", f"err <= {2 * quality_tolerance:g} px"
        return "#dc2626", f"err > {2 * quality_tolerance:g} px"

    def _bound_runs(traj):
        states = getattr(traj, "states", None)
        if not states:
            return []

        runs = []
        current_start = None
        for frame, state in zip(traj.frames(), states):
            is_bound = state == "bound"
            if is_bound and current_start is None:
                current_start = frame
            elif not is_bound and current_start is not None:
                runs.append((current_start, frame))
                current_start = None

        if current_start is not None:
            runs.append((current_start, traj.end_frame + 1))

        return runs

    gt_by_id = {
        traj.id: traj
        for traj in trajectories_GT
        if traj.id is not None and traj.length() > 0
    }

    gt_types = {_particle_type(traj) for traj in gt_by_id.values()}
    panel_types = (_ordered_types(gt_types) or ["all"]) if separate_types else ["all"]

    assigned_tracks = []
    unassigned_tracks = []

    for traj in trajectories_new:
        if traj.length() == 0:
            continue

        if traj.id is None or traj.id == -1 or traj.id not in gt_by_id:
            unassigned_tracks.append(traj)
            continue

        gt_traj = gt_by_id[traj.id]
        distances = _overlap_distances(traj, gt_traj)
        mean_error = float(np.mean(distances)) if distances else np.inf
        overlap_length = len(distances)
        gt_coverage = overlap_length / gt_traj.length() if gt_traj.length() > 0 else 0.0
        track_coverage = overlap_length / traj.length() if traj.length() > 0 else 0.0

        assigned_tracks.append({
            "traj": traj,
            "gt": gt_traj,
            "type": _particle_type(gt_traj),
            "mean_error": mean_error,
            "gt_coverage": gt_coverage,
            "track_coverage": track_coverage,
        })

    include_unassigned = show_unassigned and len(unassigned_tracks) > 0
    n_panels = len(panel_types) + int(include_unassigned)

    if figsize is None:
        total_rows = len(trajectories_GT) + (1 if include_unassigned else 0)
        figsize = (14, max(4.0, 1.0 + 0.55 * total_rows))

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [1] * n_panels},
    )
    axes = axes.ravel()

    max_frame = 0
    for traj in list(trajectories_GT) + list(trajectories_new):
        if traj.length() > 0:
            max_frame = max(max_frame, traj.end_frame + 1)

    legend_handles = [
        Line2D([0], [0], color="#cbd5e1", linewidth=8, label="Ground truth"),
        Line2D([0], [0], color="#2ca25f", linewidth=3, linestyle="--",
               label=f"Tracked, err <= {quality_tolerance:g} px"),
        Line2D([0], [0], color="#f59e0b", linewidth=3, linestyle="--",
               label=f"Tracked, err <= {2 * quality_tolerance:g} px"),
        Line2D([0], [0], color="#dc2626", linewidth=3, linestyle="--",
               label=f"Tracked, err > {2 * quality_tolerance:g} px"),
        Line2D([0], [0], color="#7c3aed", linewidth=2,
               label="GT bound state"),
    ]

    for ax_idx, panel_type in enumerate(panel_types):
        ax = axes[ax_idx]

        if panel_type == "all":
            gt_panel = list(gt_by_id.values())
            track_panel = assigned_tracks
            panel_title = "All particles"
        else:
            gt_panel = [
                traj for traj in gt_by_id.values()
                if _particle_type(traj) == panel_type
            ]
            track_panel = [
                row for row in assigned_tracks
                if row["type"] == panel_type
            ]
            panel_title = panel_type.capitalize() + "s"

        gt_panel = sorted(gt_panel, key=lambda traj: traj.id)
        y_by_id = {traj.id: i for i, traj in enumerate(gt_panel)}

        good_ids = {
            row["gt"].id
            for row in track_panel
            if row["mean_error"] <= quality_tolerance
        }
        good_rows = [
            row for row in track_panel
            if row["mean_error"] <= quality_tolerance
        ]

        mean_error = (
            np.mean([row["mean_error"] for row in good_rows])
            if good_rows else np.nan
        )
        mean_coverage = (
            np.mean([row["gt_coverage"] for row in good_rows])
            if good_rows else np.nan
        )

        stats = (
            f"recovered {len(good_ids)}/{len(gt_panel)}"
            if len(gt_panel) > 0 else "no GT"
        )
        if np.isfinite(mean_error):
            stats += f", mean err {mean_error:.2f} px"
        if np.isfinite(mean_coverage):
            stats += f", mean coverage {100 * mean_coverage:.0f}%"

        for gt_traj in gt_panel:
            y = y_by_id[gt_traj.id]
            ax.hlines(
                y,
                gt_traj.start_frame,
                gt_traj.end_frame + 1,
                colors="#cbd5e1",
                linewidth=8,
                zorder=1,
            )

            if show_bound_states:
                for start, end in _bound_runs(gt_traj):
                    ax.hlines(
                        y - 0.24,
                        start,
                        end,
                        colors="#7c3aed",
                        linewidth=2,
                        alpha=0.75,
                        zorder=2,
                    )

        duplicate_count = {}
        for row in track_panel:
            traj = row["traj"]
            gt_traj = row["gt"]
            y = y_by_id.get(gt_traj.id)
            if y is None:
                continue

            key = (panel_type, gt_traj.id)
            duplicate_idx = duplicate_count.get(key, 0)
            duplicate_count[key] = duplicate_idx + 1
            y_offset = 0.18 + 0.08 * min(duplicate_idx, 3)

            color, _ = _quality(row["mean_error"])
            ax.hlines(
                y + y_offset,
                traj.start_frame,
                traj.end_frame + 1,
                colors=color,
                linestyles="--",
                linewidth=3,
                zorder=3,
            )
            ax.plot(
                [traj.start_frame, traj.end_frame + 1],
                [y + y_offset, y + y_offset],
                marker="|",
                color=color,
                linestyle="None",
                markersize=8,
                zorder=4,
            )

            if show_labels:
                label = (
                    f"{row['mean_error']:.1f}px, "
                    f"{100 * row['gt_coverage']:.0f}%"
                )
                ax.text(
                    traj.end_frame + 1,
                    y + y_offset,
                    label,
                    va="center",
                    ha="left",
                    fontsize=8,
                    color=color,
                )

        tick_labels = [
            f"{_particle_type(traj)[0].upper()}{traj.id}"
            for traj in gt_panel
        ]

        ax.set_yticks(range(len(gt_panel)))
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(-0.6, max(len(gt_panel) - 0.2, 0.6))
        ax.set_ylabel("GT id")
        ax.set_title(f"{panel_title}: {stats}", loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0.12)

    if include_unassigned:
        ax = axes[-1]
        unassigned_tracks = sorted(
            unassigned_tracks,
            key=lambda traj: (traj.start_frame, traj.end_frame),
        )
        for idx, traj in enumerate(unassigned_tracks):
            ax.hlines(
                idx,
                traj.start_frame,
                traj.end_frame + 1,
                colors="#6b7280",
                linestyles=":",
                linewidth=2.5,
            )
        ax.set_yticks(range(len(unassigned_tracks)))
        ax.set_yticklabels([f"U{i}" for i in range(len(unassigned_tracks))])
        ax.set_ylim(-0.6, max(len(unassigned_tracks) - 0.2, 0.6))
        ax.set_ylabel("Track")
        ax.set_title(f"Unassigned tracks: {len(unassigned_tracks)}", loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)

    axes[-1].set_xlabel("Frame")
    for ax in axes:
        ax.set_xlim(0, max(max_frame, 1))

    if title is None:
        title = "Tracking quality by particle type"
    fig.suptitle(title, y=0.995)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(len(legend_handles), 5),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)

    plt.show()
    if return_fig:
        return fig, axes

    return None    

def GT_linear_visualizer(
    trajectories_GT_or_sim,
    title=None,
    figsize=None,
    save_path=None,
    return_fig=False,
    show_event_labels=True,
    show_bound_state_runs=True,
):
    """
    Enhanced visualization of GT ligand–receptor interactions with:
    - hierarchical layout (ligands grouped under receptors they bind)
    - receptor-based color palette
    - faded unbound segments, highlighted bound segments
    - curved ligand–receptor connectors
    - global interaction overview bar
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _as_trajectories(obj):
        if hasattr(obj, "trajectories_GT") and getattr(obj, "trajectories_GT") is not None:
            return list(obj.trajectories_GT)
        return list(obj)

    def _ptype(traj):
        return getattr(traj, "particle_type", "particle") or "particle"

    def _label(traj):
        prefix = {"receptor": "R", "ligand": "L"}.get(_ptype(traj), _ptype(traj)[0].upper())
        return f"{prefix}{traj.id}"

    def _bound_runs(traj):
        states = list(getattr(traj, "states", []) or [])
        bound_to = list(getattr(traj, "bound_to", []) or [])
        if not states or not bound_to:
            return []

        runs = []
        current_start = None
        current_partner = None
        prev_frame = None

        for frame, state, partner in zip(traj.frames(), states, bound_to):
            is_bound = state == "bound" and partner is not None

            if is_bound:
                if current_start is None:
                    current_start = frame
                    current_partner = partner
                elif partner != current_partner:
                    runs.append((current_start, prev_frame, current_partner))
                    current_start = frame
                    current_partner = partner
            elif current_start is not None:
                runs.append((current_start, prev_frame, current_partner))
                current_start = None
                current_partner = None

            prev_frame = frame

        if current_start is not None:
            runs.append((current_start, prev_frame, current_partner))

        return runs

    def curved_link(ax, x, y1, y2, color):
        """Draw a smooth curved connector between ligand and receptor."""
        con = FancyArrowPatch(
            (x, y1), (x, y2),
            connectionstyle="arc3,rad=0.35",
            arrowstyle="-",
            linewidth=1.2,
            color=color,
            alpha=0.35,
            zorder=2,
        )
        ax.add_patch(con)

    # ------------------------------------------------------------
    # Collect trajectories
    # ------------------------------------------------------------
    trajectories = [t for t in _as_trajectories(trajectories_GT_or_sim) if t.length() > 0]
    if not trajectories:
        raise ValueError("No GT trajectories available")

    receptors = sorted([t for t in trajectories if _ptype(t) == "receptor"], key=lambda t: t.id)
    ligands = sorted([t for t in trajectories if _ptype(t) == "ligand"], key=lambda t: t.id)
    others = sorted([t for t in trajectories if _ptype(t) not in {"receptor", "ligand"}],
                    key=lambda t: (_ptype(t), t.id))

    traj_by_id = {t.id: t for t in trajectories}

    # ------------------------------------------------------------
    # Build interaction list
    # ------------------------------------------------------------
    events = []
    for lig in ligands:
        for start, end, rid in _bound_runs(lig):
            if rid in traj_by_id:
                events.append({
                    "ligand": lig,
                    "receptor": traj_by_id[rid],
                    "start": start,
                    "end": end,
                })

    # ------------------------------------------------------------
    # Hierarchical layout: group ligands under receptors they bind
    # ------------------------------------------------------------
    receptor_to_ligands = {r.id: [] for r in receptors}
    unbound_ligands = []

    for lig in ligands:
        partners = {e["receptor"].id for e in events if e["ligand"] is lig}
        if len(partners) == 1:
            receptor_to_ligands[list(partners)[0]].append(lig)
        else:
            unbound_ligands.append(lig)

    # ------------------------------------------------------------
    # Assign y-positions
    # ------------------------------------------------------------
    y_positions = {}
    y_labels = []
    current_y = 0.0
    row_step = 1.0

    # receptors + their ligands
    for r in receptors:
        y_positions[r.id] = current_y
        y_labels.append(_label(r))
        current_y += row_step

        for lig in receptor_to_ligands[r.id]:
            y_positions[lig.id] = current_y
            y_labels.append("  " + _label(lig))  # indent
            current_y += row_step

    # unbound ligands
    if unbound_ligands:
        current_y += 0.5
        for lig in unbound_ligands:
            y_positions[lig.id] = current_y
            y_labels.append(_label(lig))
            current_y += row_step

    # others
    if others:
        current_y += 0.5
        for t in others:
            y_positions[t.id] = current_y
            y_labels.append(_label(t))
            current_y += row_step

    # ------------------------------------------------------------
    # Color palette: receptor-based
    # ------------------------------------------------------------
    cmap = plt.cm.get_cmap("tab10", max(1, len(receptors)))
    receptor_colors = {r.id: cmap(i) for i, r in enumerate(receptors)}

    def ligand_color(lig, receptor_id):
        base = np.array(receptor_colors[receptor_id])
        return base * 0.55 + 0.45  # lighter shade

    # ------------------------------------------------------------
    # Determine max frame
    # ------------------------------------------------------------
    max_frame = max(t.end_frame for t in trajectories) + 1

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    if figsize is None:
        figsize = (16, max(4.5, 0.5 * len(trajectories) + 2))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f8fafc")

    # ------------------------------------------------------------
    # Draw unbound segments (faded)
    # ------------------------------------------------------------
    for t in trajectories:
        y = y_positions[t.id]
        ax.hlines(
            y,
            t.start_frame,
            t.end_frame + 1,
            color="#cbd5e1",
            linewidth=2,
            alpha=0.25,
            zorder=1,
        )

    # ------------------------------------------------------------
    # Draw bound intervals + connectors
    # ------------------------------------------------------------
    if show_bound_state_runs:
        for e in events:
            lig = e["ligand"]
            rec = e["receptor"]
            start, end = e["start"], e["end"]
            x0, x1 = start, end + 1
            xm = 0.5 * (x0 + x1)

            yL = y_positions[lig.id]
            yR = y_positions[rec.id]

            col = ligand_color(lig, rec.id)

            # highlight ligand interval
            ax.hlines(
                yL, x0, x1,
                color=col,
                linewidth=4,
                alpha=0.95,
                zorder=3,
            )

            # highlight receptor interval (lighter)
            ax.hlines(
                yR, x0, x1,
                color=col,
                linewidth=4,
                alpha=0.35,
                zorder=2,
            )

            # curved connector
            curved_link(ax, xm, yL, yR, col)

            # label
            if show_event_labels:
                ax.text(
                    xm, min(yL, yR) - 0.12,
                    f"L{lig.id}→R{rec.id}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=col,
                    alpha=0.9,
                    zorder=4,
                )

    # ------------------------------------------------------------
    # Global interaction overview bar
    # ------------------------------------------------------------
    interaction_count = np.zeros(max_frame, dtype=int)
    for e in events:
        interaction_count[e["start"]:e["end"] + 1] += 1

    ax2 = ax.inset_axes([0, 1.01, 1, 0.06])
    ax2.plot(interaction_count, color="#475569", linewidth=1.2)
    ax2.fill_between(range(max_frame), interaction_count, color="#94a3b8", alpha=0.3)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Interaction density", fontsize=8, pad=1)

    # ------------------------------------------------------------
    # Final formatting
    # ------------------------------------------------------------
    ax.set_yticks([y_positions[t.id] for t in trajectories])
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Frame")
    ax.set_xlim(0, max_frame)
    ax.set_ylim(current_y - 0.4, -0.7)
    ax.grid(axis="x", alpha=0.22)

    if title is None:
        title = "GT ligand–receptor interaction timeline"
    fig.suptitle(title, y=0.995)

    # Legend
    legend_handles = [
        Line2D([0], [0], color="#cbd5e1", linewidth=2, alpha=0.25, label="Unbound trajectory"),
        Line2D([0], [0], color="#7c3aed", linewidth=4, label="Bound interval"),
        Line2D([0], [0], color="#7c3aed", linewidth=1.2, alpha=0.35, label="Ligand–receptor link"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)

    plt.show()
    if return_fig:
        return fig, ax
    return None

def plot_coverage_error_tradeoff(summary_df, save_path=None):
    """
    Scatter/bubble plot:
    x = mean error,
    y = coverage
    """
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    labels = summary_df["method_label"].tolist()

    x = summary_df["mean_error_px_mean"].to_numpy(dtype=float)
    y = summary_df["coverage_pct_mean"].to_numpy(dtype=float)
    n_tracks = summary_df["n_tracks_mean"].to_numpy(dtype=float)

    xerr = np.nan_to_num(summary_df["mean_error_px_ci95"].to_numpy(dtype=float), nan=0.0)
    yerr = np.nan_to_num(summary_df["coverage_pct_ci95"].to_numpy(dtype=float), nan=0.0)

    sizes = np.full_like(n_tracks, 500.0)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    for i, label in enumerate(labels):
        ax.errorbar(
            x[i],
            y[i],
            xerr=xerr[i],
            yerr=yerr[i],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
            alpha=0.6,
            zorder=1,
        )

        ax.scatter(
            x[i],
            y[i],
            s=sizes[i],
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            label=f"{label} (tracks={n_tracks[i]:.1f})",
            zorder=2,
        )

        ax.annotate(
            label,
            (x[i], y[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Mean localization error (px)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage-error trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig