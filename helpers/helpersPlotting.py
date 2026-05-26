




def plot_ligand_receptor_frame(sim, frame_id=0, show_ids=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = np.percentile(sim.frames, 99.8)
    ax.imshow(sim.frames[frame_id], cmap="gray", vmin=0, vmax=vmax)

    for traj in sim.trajectories_GT:
        pos = traj.get_position_at_frame(frame_id)
        if pos is None:
            continue

        row, col = pos
        is_receptor = traj.particle_type == "receptor"
        is_bound = traj.get_state_at_frame(frame_id) == "bound"

        ax.scatter(
            col,
            row,
            s=75 if is_receptor else 26,
            c="tab:red" if is_receptor else "tab:cyan",
            marker="s" if is_receptor else "o",
            edgecolors="yellow" if is_bound else "black",
            linewidths=1.2 if is_bound else 0.4,
            alpha=0.85,
        )

        if show_ids:
            ax.text(col + 1, row + 1, str(traj.id), color="white", fontsize=8)

    ax.set_title(f"Ligand-receptor frame {frame_id}")
    ax.axis("off")
    plt.show()

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


def plot_gt_frame(ax, sim, frame_id):
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
            c=TYPE_COLORS.get(gt_type, "white"),
            marker="s" if gt_type == "receptor" else "o",
            edgecolors="yellow" if is_bound else "black",
            linewidths=1.2 if is_bound else 0.4,
            alpha=0.9,
        )
    ax.set_title(f"GT frame {frame_id}: ligands, receptors, bound states", fontsize=10)
    ax.axis("off")


def plot_tracks_on_axis(ax, frames, trajectories, trajectories_GT, frame_id, title):
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
        color = TYPE_COLORS.get(particle_type(gt_traj), "#6b7280") if gt_traj is not None else "#6b7280"
        ax.plot(positions[:, 1], positions[:, 0], color=color, linewidth=1.7, alpha=0.9)
        ax.plot(positions[-1, 1], positions[-1, 0], marker="+", color=color, markersize=6)

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_coverage_panel(ax, summary_table):
    width = 0.35
    x = np.arange(2)
    populations = ["ligand", "receptor"]
    for idx, method in enumerate(METHOD_ORDER):
        values = [
            summary_table.loc[
                (summary_table["method"] == method) & (summary_table["population"] == pop),
                "mean_coverage_pct",
            ].iloc[0]
            for pop in populations
        ]
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([pop.capitalize() for pop in populations])
    ax.set_ylabel("GT frame coverage within 3 px (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Coverage by particle type", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def plot_error_panel(ax, regular_metrics, masked_metrics):
    positions = [0.8, 1.2, 1.8, 2.2]
    labels = ["Ligand", "", "Receptor", ""]
    data = []
    colors = []
    for pop in ["ligand", "receptor"]:
        for method, metrics in [("Regular full", regular_metrics), ("Masked static-first", masked_metrics)]:
            errors = metrics["errors"]
            values = errors.loc[errors["type"] == pop, "error_px"].dropna().to_numpy()
            data.append(values if len(values) else np.asarray([np.nan]))
            colors.append(METHOD_COLORS[method])

    bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(["Ligand", "Receptor"])
    ax.set_ylabel("Position error (px)")
    ax.set_title("Localization error distribution", fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def plot_state_panel(ax, state_summary):
    ligand_state = state_summary[state_summary["type"] == "ligand"].copy()
    states = [state for state in ["free", "bound"] if state in set(ligand_state["state"])]
    if not states:
        ax.text(0.5, 0.5, "No ligand state data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    width = 0.35
    x = np.arange(len(states))
    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for state in states:
            row = ligand_state[(ligand_state["method"] == method) & (ligand_state["state"] == state)]
            values.append(float(row["coverage_pct"].iloc[0]) if len(row) else 0.0)
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([state.capitalize() for state in states])
    ax.set_ylabel("Ligand coverage within 3 px (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Ligand recovery by binding state", fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def plot_peak_count_panel(ax, peaks_full, peaks_masked):
    full_counts = np.asarray([len(frame_peaks) for frame_peaks in peaks_full], dtype=float)
    masked_counts = np.asarray([len(frame_peaks) for frame_peaks in peaks_masked], dtype=float)
    frames_axis = np.arange(len(full_counts))
    ax.plot(frames_axis, full_counts, color=METHOD_COLORS["Regular full"], linewidth=1.8, label="All peaks")
    ax.plot(frames_axis, masked_counts, color=METHOD_COLORS["Masked static-first"], linewidth=1.8, label="After static mask")
    ax.fill_between(frames_axis, masked_counts, full_counts, color="#9ca3af", alpha=0.25, label="Removed")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Localized peaks")
    ax.set_title("Mask effect on detections", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)


def plot_ligand_receptor_comparison_figure():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    plot_gt_frame(axes[0], sim, FRAME_FOR_FIGURE)
    plot_tracks_on_axis(
        axes[1], frames, regular_trajs, trajectories_GT, FRAME_FOR_FIGURE,
        "Regular full tracking",
    )
    plot_tracks_on_axis(
        axes[2], frames, masked_combined_trajs, trajectories_GT, FRAME_FOR_FIGURE,
        "Masked static-first tracking",
    )
    plot_coverage_panel(axes[3], summary_table)
    plot_error_panel(axes[4], regular_metrics, masked_metrics)
    plot_state_panel(axes[5], state_summary)

    for label, ax in zip("ABCDEF", axes):
        add_panel_label(ax, label)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TYPE_COLORS["ligand"], markeredgecolor="black", label="Ligand"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TYPE_COLORS["receptor"], markeredgecolor="black", label="Receptor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="yellow", label="Bound state"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Regular full tracking vs masked static-first tracking", y=1.04, fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT_FIGURE_BASE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(REPORT_FIGURE_BASE.with_suffix(".pdf"), bbox_inches="tight")
    return fig