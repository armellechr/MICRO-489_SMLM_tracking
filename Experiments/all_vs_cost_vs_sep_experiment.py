from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Project setup
# =============================================================================

def _project_root() -> Path:
    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve()

    candidates = [
        cwd,
        cwd.parent,
        here.parent.parent,
        cwd / "MICRO-489_SMLM_tracking",
    ]

    for candidate in candidates:
        if (candidate / "Helpers").exists():
            return candidate

    raise RuntimeError(f"Could not find project root from {cwd}")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from Helpers.helpersAssignment import (  # noqa: E402
    LengthDifferenceTerm,
    MeanPositionDistanceTerm,
    StartFrameDifferenceTerm,
    TrajToTraj,
    assign_trajectories,
)
from Helpers.helpersSimulation import LigandReceptorSimulator, Simulator  # noqa: E402
from Helpers.helpersTracking import (  # noqa: E402
    DistanceTerm,
    IntensityTerm,
    PeakToPeak,
    SigmaTerm,
    bridge_mobile_fragments_through_static_anchors,
    extract_peaks,
    remove_static_peaks,
    track_from_peaks,
)


# =============================================================================
# Output paths
# =============================================================================

OUT_DIR = PROJECT_ROOT / "Results"
OUT_DIR.mkdir(exist_ok=True)

RESULTS_PATH = OUT_DIR / "AllvsCostvsSep_hota_results.csv"
SUMMARY_PATH = OUT_DIR / "AllvsCostvsSep_hota_summary.csv"
FIGURE_PATH = OUT_DIR / "AllvsCostvsSep_hota_summary.png"

LR_RESULTS_PATH = OUT_DIR / "AllvsCostvsSep_ligand_receptor_hota_results.csv"
LR_SUMMARY_PATH = OUT_DIR / "AllvsCostvsSep_ligand_receptor_hota_summary.csv"
LR_FIGURE_PATH = OUT_DIR / "AllvsCostvsSep_ligand_receptor_hota_summary.png"

RECOMMENDATIONS_PATH = OUT_DIR / "CostExperiment_recommendations.csv"


# =============================================================================
# Method names
# =============================================================================

METHOD_ORDER = ["all_distance", "all_enhanced", "separated_distance"]

METHOD_LABELS = {
    "all_distance": "All - distance",
    "all_enhanced": "All - enhanced cost",
    "separated_distance": "Separated - distance",
}


# =============================================================================
# Experiment configuration
# =============================================================================

SEEDS = list(range(2))

D_LIST = [
    (0.02, 0.16),
    (0.08, 0.14),
    (0.25, 0.14),
    (0.75, 0.14),
    (2.00, 0.14),
    (4.50, 0.14),
    (8.00, 0.14),
]

SIMULATION_CONFIG = {
    "nparticles": 20,
    "nframes": 100,
    "nposframe": 10,
    "D_list": D_LIST,
    "dt": 1.0,
    "anisotropy_ratio_range": (0.75, 1.0),
    "theta_range": (0, np.pi),
    "frame_size": (128, 128),
    "intensity_mean": 600,
    "intensity_std": 150,
    "sigma_mean": 1.0,
    "sigma_std": 0.35,
    "boundary_margin": 14,
}

IMAGE_CONFIG = {
    "particle_intensity": [600, 150],
    "particle_sigma": [1.0, 0.35],
    "background_intensity": [150, 50],
    "poisson_noise": 100,
    "output_size": 128,
    "upsampling_factor": 3,
    "resolution": 100e-9,
    "trajectory_unit": -1,
    "invert_y": True,
    "use_trajectory_sigma": True,
}

TRACKING_KWARGS = {
    "mode": "localization",
    "detection_threshold": 400,
    "r_squared_threshold": 0.35,
    "max_distance": 10.0,
    "min_length": 5,
    "max_gap": 1,
    "algo_peak2peak": "hungarian",
    "algo_traj2traj": "hungarian",
}

STATIC_TRACKING_KWARGS = {
    "max_distance": 1.2,
    "min_length": 15,
    "max_gap": 0,
    "remove_tolerance": 2.5,
}

LR_SEEDS = list(range(2))

LR_SIMULATION_CONFIG = {
    "n_ligands": 10,
    "n_receptors": 5,
    "nframes": 50,
    "nposframe": 10,
    "dt": 1.0,
    "ligand_D": [(3.5, 0.35), (5.0, 0.45), (7.0, 0.20)],
    "receptor_D": [(0.0, 0.35), (0.03, 0.65)],
    "frame_size": (128, 128),
    "boundary_margin": 12,
    "binding_radius": 5.0,
    "kon": 1.0,
    "koff": 0.06,
    "allow_multiple_ligands_per_receptor": False,
    "bound_position_noise": 0.15,
    "ligand_intensity_mean": 620,
    "ligand_intensity_std": 120,
    "receptor_intensity_mean": 760,
    "receptor_intensity_std": 150,
    "ligand_sigma_mean": 0.85,
    "ligand_sigma_std": 0.08,
    "receptor_sigma_mean": 1.15,
    "receptor_sigma_std": 0.12,
    "reflect_boundaries": True,
}

LR_IMAGE_CONFIG = {
    "particle_intensity": [680, 160],
    "particle_sigma": [1.0, 0.2],
    "particle_type_props": {
        "ligand": {
            "particle_intensity": [620, 120],
            "particle_sigma": [0.85, 0.08],
        },
        "receptor": {
            "particle_intensity": [760, 150],
            "particle_sigma": [1.15, 0.12],
        },
    },
    "use_trajectory_sigma": True,
    "background_intensity": [100, 25],
    "poisson_noise": 100,
    "output_size": 128,
    "upsampling_factor": 3,
    "resolution": 100e-9,
    "trajectory_unit": -1,
    "invert_y": True,
}

LR_TRACKING_KWARGS = {
    "mode": "localization",
    "detection_threshold": 330,
    "r_squared_threshold": 0.35,
    "max_distance": 14.0,
    "min_length": 2,
    "max_gap": 5,
    "stitch_fragments": True,
    "stitch_max_gap": 8,
    "stitch_base_distance": 12.0,
    "stitch_kwargs": {"max_link_cost": 2.5, "gap_weight": 0.05},
    "algo_peak2peak": "hungarian",
    "algo_traj2traj": "hungarian",
}

LR_STATIC_TRACKING_KWARGS = {
    "detection_threshold": 430,
    "r_squared_threshold": 0.45,
    "max_distance": 1.6,
    "min_length": 20,
    "max_gap": 0,
    "remove_tolerance": 3.0,
}

LR_BRIDGE_KWARGS = {
    "max_gap": 14,
    "anchor_radius": 2 * LR_SIMULATION_CONFIG["binding_radius"],
    "max_link_cost": 2.5,
    "gap_weight": 0.04,
    "fill_bound_frames": True,
    "return_diagnostics": False,
    "verbose": False,
}


# =============================================================================
# Cost functions
# =============================================================================

PEAK_NORMS = {
    "distance": TRACKING_KWARGS["max_distance"],
    "intensity": IMAGE_CONFIG["particle_intensity"][0],
    "sigma": 0.75,
}

LR_PEAK_NORMS = {
    "distance": LR_TRACKING_KWARGS["max_distance"],
    "intensity": LR_IMAGE_CONFIG["particle_intensity"][0],
    "sigma": 0.75,
}

TRAJ_NORMS = {
    "position": TRACKING_KWARGS["max_distance"],
    "length": 10.0,
    "start_frame": 5.0,
}

TRUTH_MATCH_TOLERANCE_PX = 3.0

FALLBACK_PEAK_SPEC = {
    "name": "dist_int_60_40",
    "distance": 0.60,
    "intensity": 0.40,
    "sigma": 0.00,
}

FALLBACK_TRAJ_SPEC = {
    "name": "equal_all",
    "position": 1 / 3,
    "length": 1 / 3,
    "start_frame": 1 / 3,
}


def _value_or_default(row, column, default):
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def load_recommended_specs(path: Path = RECOMMENDATIONS_PATH):
    if not path.exists():
        return copy.deepcopy(FALLBACK_PEAK_SPEC), copy.deepcopy(FALLBACK_TRAJ_SPEC)

    recommendations = pd.read_csv(path)
    peak_rows = recommendations[recommendations["stage"] == "peak2peak"]
    traj_rows = recommendations[recommendations["stage"] == "traj2traj"]

    if len(peak_rows):
        peak_row = peak_rows.iloc[0]
        peak_spec = {
            "name": peak_row.get("recommended_config", FALLBACK_PEAK_SPEC["name"]),
            "distance": _value_or_default(peak_row, "distance", FALLBACK_PEAK_SPEC["distance"]),
            "intensity": _value_or_default(peak_row, "intensity", FALLBACK_PEAK_SPEC["intensity"]),
            "sigma": _value_or_default(peak_row, "sigma", FALLBACK_PEAK_SPEC["sigma"]),
        }
    else:
        peak_spec = copy.deepcopy(FALLBACK_PEAK_SPEC)

    if len(traj_rows):
        traj_row = traj_rows.iloc[0]
        traj_spec = {
            "name": traj_row.get("recommended_config", FALLBACK_TRAJ_SPEC["name"]),
            "position": _value_or_default(traj_row, "position", FALLBACK_TRAJ_SPEC["position"]),
            "length": _value_or_default(traj_row, "length", FALLBACK_TRAJ_SPEC["length"]),
            "start_frame": _value_or_default(
                traj_row,
                "start_frame",
                FALLBACK_TRAJ_SPEC["start_frame"],
            ),
        }
    else:
        traj_spec = copy.deepcopy(FALLBACK_TRAJ_SPEC)

    return peak_spec, traj_spec


RECOMMENDED_PEAK_SPEC, RECOMMENDED_TRAJ_SPEC = load_recommended_specs()


def build_peak_cost(spec, peak_norms=None):
    peak_norms = peak_norms or PEAK_NORMS
    terms = {}
    if spec.get("distance", 0) > 0:
        terms["distance"] = DistanceTerm(weight=float(spec["distance"]), norm=peak_norms["distance"])
    if spec.get("intensity", 0) > 0:
        terms["intensity"] = IntensityTerm(weight=float(spec["intensity"]), norm=peak_norms["intensity"])
    if spec.get("sigma", 0) > 0:
        terms["sigma"] = SigmaTerm(weight=float(spec["sigma"]), norm=peak_norms["sigma"])
    return PeakToPeak(terms=terms)


def build_distance_peak_cost(norm=None):
    return PeakToPeak(
        terms={
            "distance": DistanceTerm(
                weight=1.0,
                norm=float(norm or TRACKING_KWARGS["max_distance"]),
            )
        }
    )


def build_traj_cost(spec):
    terms = {}
    if spec.get("position", 0) > 0:
        terms["position"] = MeanPositionDistanceTerm(
            weight=float(spec["position"]),
            norm=TRAJ_NORMS["position"],
            invalid_cost=1e6,
        )
    if spec.get("length", 0) > 0:
        terms["length"] = LengthDifferenceTerm(
            weight=float(spec["length"]),
            norm=TRAJ_NORMS["length"],
        )
    if spec.get("start_frame", 0) > 0:
        terms["start_frame"] = StartFrameDifferenceTerm(
            weight=float(spec["start_frame"]),
            norm=TRAJ_NORMS["start_frame"],
        )
    return TrajToTraj(terms=terms)


# =============================================================================
# Generic metric helpers
# =============================================================================

def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def finite_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def valid_gt_id(gid, gt_by_id):
    return gid is not None and gid != -1 and gid in gt_by_id


def detections_by_frame(trajectories, identity="track"):
    by_frame = {}

    for track_idx, traj in enumerate(trajectories):
        if traj.length() == 0:
            continue

        det_id = track_idx if identity == "track" else traj.id

        if identity == "gt" and (det_id is None or det_id == -1):
            continue

        for frame in traj.frames():
            pos = traj.get_position_at_frame(frame)
            if pos is None:
                continue

            by_frame.setdefault(int(frame), []).append({
                "id": det_id,
                "pos": np.asarray(pos, dtype=float),
                "particle_type": getattr(traj, "particle_type", None),
            })

    return by_frame


def match_detections_by_frame(trajectories, trajectories_gt, tolerance_px=TRUTH_MATCH_TOLERANCE_PX):
    pred_by_frame = detections_by_frame(trajectories, identity="track")
    gt_by_frame = detections_by_frame(trajectories_gt, identity="gt")

    total_pred = sum(len(v) for v in pred_by_frame.values())
    total_gt = sum(len(v) for v in gt_by_frame.values())
    match_rows = []

    for frame in sorted(set(pred_by_frame) | set(gt_by_frame)):
        preds = pred_by_frame.get(frame, [])
        gts = gt_by_frame.get(frame, [])

        if not preds or not gts:
            continue

        cost = np.empty((len(preds), len(gts)), dtype=float)
        for i, pred in enumerate(preds):
            for j, gt in enumerate(gts):
                cost[i, j] = float(np.linalg.norm(pred["pos"] - gt["pos"]))

        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind):
            error_px = float(cost[i, j])
            if error_px <= tolerance_px:
                match_rows.append({
                    "frame": int(frame),
                    "gt_id": gts[j]["id"],
                    "track_id": preds[i]["id"],
                    "error_px": error_px,
                    "gt_type": gts[j].get("particle_type", None),
                })

    return pd.DataFrame(match_rows), total_pred, total_gt


def hota_like_metrics(trajectories, trajectories_gt, tolerance_px=TRUTH_MATCH_TOLERANCE_PX):
    matches, total_pred, total_gt = match_detections_by_frame(
        trajectories,
        trajectories_gt,
        tolerance_px=tolerance_px,
    )

    tp = int(len(matches))
    fp = int(total_pred - tp)
    fn = int(total_gt - tp)

    deta = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan

    if tp:
        pair_counts = Counter(zip(matches["gt_id"], matches["track_id"]))
        gt_counts = Counter(matches["gt_id"])
        track_counts = Counter(matches["track_id"])
        ass_scores = []

        for gt_id, track_id in zip(matches["gt_id"], matches["track_id"]):
            tpa = pair_counts[(gt_id, track_id)]
            fna = gt_counts[gt_id] - tpa
            fpa = track_counts[track_id] - tpa
            ass_scores.append(tpa / (tpa + fna + fpa))

        assa = finite_mean(ass_scores)
    else:
        assa = np.nan

    hota = (
        float(np.sqrt(deta * assa))
        if np.isfinite(deta) and np.isfinite(assa)
        else np.nan
    )

    id_precision = tp / (tp + fp) if (tp + fp) else np.nan
    id_recall = tp / (tp + fn) if (tp + fn) else np.nan
    idf1 = (
        2 * id_precision * id_recall / (id_precision + id_recall)
        if np.isfinite(id_precision)
        and np.isfinite(id_recall)
        and (id_precision + id_recall) > 0
        else np.nan
    )

    id_switches = 0
    gt_ids = sorted(matches["gt_id"].unique()) if tp else []
    for gt_id in gt_ids:
        gt_matches = matches[matches["gt_id"] == gt_id].sort_values("frame")
        previous_track = None
        for track_id in gt_matches["track_id"].tolist():
            if previous_track is not None and track_id != previous_track:
                id_switches += 1
            previous_track = track_id

    mota = 1.0 - (fn + fp + id_switches) / total_gt if total_gt else np.nan
    motp_px = finite_mean(matches["error_px"]) if tp else np.nan

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "deta": deta,
        "assa": assa,
        "hota": hota,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "idf1": idf1,
        "mota": mota,
        "motp_px": motp_px,
        "id_switches": int(id_switches),
        "matches": matches,
    }


def edge_counter(trajectories, require_valid_gt=True, gt_by_id=None):
    edges = Counter()

    for traj in trajectories:
        gid = traj.id

        if require_valid_gt and not valid_gt_id(gid, gt_by_id):
            continue

        frames = sorted(traj.frames())
        for f0, f1 in zip(frames[:-1], frames[1:]):
            if f1 == f0 + 1:
                edges[(int(gid), int(f0), int(f1))] += 1

    return edges


def precision_recall_f1(pred_edges, gt_edges):
    tp = sum(min(count, gt_edges.get(edge, 0)) for edge, count in pred_edges.items())
    fp = sum(pred_edges.values()) - tp
    fn = sum(gt_edges.values()) - tp

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and (precision + recall) > 0
        else np.nan
    )

    return precision, recall, f1, tp, fp, fn


def assigned_position_errors(trajectories, gt_by_id):
    errors = []

    for traj in trajectories:
        if not valid_gt_id(traj.id, gt_by_id):
            continue

        gt_traj = gt_by_id[traj.id]

        for frame in traj.frames():
            pos = traj.get_position_at_frame(frame)
            gt_pos = gt_traj.get_position_at_frame(frame)

            if pos is None or gt_pos is None:
                continue

            errors.append(float(np.linalg.norm(np.asarray(pos) - np.asarray(gt_pos))))

    return np.asarray(errors, dtype=float)


def fragmentation_count(gt_frames, matched_frames):
    matched_frames = set(matched_frames)

    was_matched = False
    has_been_matched = False
    n_fragments = 0

    for frame in sorted(gt_frames):
        is_matched = frame in matched_frames

        if is_matched and has_been_matched and not was_matched:
            n_fragments += 1

        if is_matched:
            has_been_matched = True

        was_matched = is_matched

    return n_fragments


def coverage_fragmentation_from_matches(matches, trajectories_gt):
    gt_by_id = {
        traj.id: traj
        for traj in trajectories_gt
        if traj.id is not None and traj.length() > 0
    }

    matched_frames_by_gt = {gid: set() for gid in gt_by_id}

    if len(matches):
        for row in matches.itertuples(index=False):
            if row.gt_id in matched_frames_by_gt:
                matched_frames_by_gt[row.gt_id].add(int(row.frame))

    coverage = []
    fragmentations = []

    for gid, gt_traj in gt_by_id.items():
        gt_frames = [
            int(frame)
            for frame in gt_traj.frames()
            if gt_traj.get_position_at_frame(frame) is not None
        ]
        matched_frames = matched_frames_by_gt[gid]

        coverage.append(len(matched_frames) / len(gt_frames) if gt_frames else np.nan)
        fragmentations.append(fragmentation_count(gt_frames, matched_frames))

    return np.asarray(coverage, dtype=float), np.asarray(fragmentations, dtype=float)


def mot_tracking_quality_row(method_name, trajectories, trajectories_gt, seed, population="all"):
    metrics = hota_like_metrics(
        trajectories,
        trajectories_gt,
        tolerance_px=TRUTH_MATCH_TOLERANCE_PX,
    )

    gt_by_id = {
        traj.id: traj
        for traj in trajectories_gt
        if traj.id is not None and traj.length() > 0
    }

    gt_edges = edge_counter(trajectories_gt, require_valid_gt=False)
    pred_edges = edge_counter(trajectories, require_valid_gt=True, gt_by_id=gt_by_id)
    link_precision, link_recall, link_f1, link_tp, link_fp, link_fn = precision_recall_f1(
        pred_edges,
        gt_edges,
    )

    coverage, fragmentations = coverage_fragmentation_from_matches(
        metrics["matches"],
        trajectories_gt,
    )

    errors = metrics["matches"]["error_px"].to_numpy(dtype=float) if len(metrics["matches"]) else np.asarray([])

    row = {
        "seed": seed,
        "method": method_name,
        "population": population,
        "method_label": METHOD_LABELS[method_name],
        "hota": metrics["hota"],
        "deta": metrics["deta"],
        "assa": metrics["assa"],
        "idf1": metrics["idf1"],
        "id_precision": metrics["id_precision"],
        "id_recall": metrics["id_recall"],
        "mota": metrics["mota"],
        "motp_px": metrics["motp_px"],
        "id_switches": metrics["id_switches"],
        "link_precision": link_precision,
        "link_recall": link_recall,
        "link_f1": link_f1,
        "link_tp": link_tp,
        "link_fp": link_fp,
        "link_fn": link_fn,
        "coverage": finite_mean(coverage),
        "coverage_pct": 100.0 * finite_mean(coverage),
        "median_error_px": finite_median(errors),
        "mean_error_px": finite_mean(errors),
        "fragmentations": int(np.nansum(fragmentations)) if len(fragmentations) else 0,
        "mean_fragmentations_per_gt": finite_mean(fragmentations),
        "n_tracks": len([traj for traj in trajectories if traj.length() > 0]),
    }

    return row


def filter_trajectories_by_population(trajectories, population):
    return [
        traj
        for traj in trajectories
        if getattr(traj, "particle_type", None) == population
        or getattr(traj, "metadata", {}).get("role", None) == population
    ]


# =============================================================================
# Homogeneous simulation pipeline
# =============================================================================

def make_simulation(seed):
    return Simulator(
        simulation_config=copy.deepcopy(SIMULATION_CONFIG),
        image_config=copy.deepcopy(IMAGE_CONFIG),
        seed=seed,
    ).run()


def extract_peaks_once(sim):
    return extract_peaks(
        sim.frames,
        mode=TRACKING_KWARGS["mode"],
        detection_threshold=TRACKING_KWARGS["detection_threshold"],
        r_squared_threshold=TRACKING_KWARGS["r_squared_threshold"],
        verbose_loc=False,
        visualization_loc=False,
    )


def run_all_particles_method(peaks, trajectories_gt, peak_cost, traj_cost):
    trajectories, _, _, _ = track_from_peaks(
        peaks,
        trajectories_gt,
        max_distance=TRACKING_KWARGS["max_distance"],
        min_length=TRACKING_KWARGS["min_length"],
        max_gap=TRACKING_KWARGS["max_gap"],
        algo_peak2peak=TRACKING_KWARGS["algo_peak2peak"],
        cost_func_peak2peak=peak_cost,
        algo_traj2traj=TRACKING_KWARGS["algo_traj2traj"],
        cost_func_traj2traj=traj_cost,
        verbose_assignment=False,
    )
    return trajectories


def run_separated_distance_method(peaks, trajectories_gt, traj_cost):
    static_peak_cost = build_distance_peak_cost(STATIC_TRACKING_KWARGS["max_distance"])
    mobile_peak_cost = build_distance_peak_cost()

    static_trajectories, _, _, _ = track_from_peaks(
        peaks,
        trajectories_gt,
        max_distance=STATIC_TRACKING_KWARGS["max_distance"],
        min_length=STATIC_TRACKING_KWARGS["min_length"],
        max_gap=STATIC_TRACKING_KWARGS["max_gap"],
        algo_peak2peak=TRACKING_KWARGS["algo_peak2peak"],
        cost_func_peak2peak=static_peak_cost,
        algo_traj2traj=TRACKING_KWARGS["algo_traj2traj"],
        cost_func_traj2traj=traj_cost,
        verbose_assignment=False,
    )

    mobile_peaks = remove_static_peaks(
        peaks,
        static_trajectories,
        tolerance=STATIC_TRACKING_KWARGS["remove_tolerance"],
        verbose=False,
    )

    mobile_trajectories, _, _, _ = track_from_peaks(
        mobile_peaks,
        trajectories_gt,
        max_distance=TRACKING_KWARGS["max_distance"],
        min_length=TRACKING_KWARGS["min_length"],
        max_gap=TRACKING_KWARGS["max_gap"],
        algo_peak2peak=TRACKING_KWARGS["algo_peak2peak"],
        cost_func_peak2peak=mobile_peak_cost,
        algo_traj2traj=TRACKING_KWARGS["algo_traj2traj"],
        cost_func_traj2traj=traj_cost,
        verbose_assignment=False,
    )

    combined = copy.deepcopy(static_trajectories) + copy.deepcopy(mobile_trajectories)

    if combined:
        combined, _, _ = assign_trajectories(
            combined,
            trajectories_gt,
            algorithm=TRACKING_KWARGS["algo_traj2traj"],
            cost_function=traj_cost,
            verbose=False,
        )

    return combined


def run_one_seed(seed, verbose=True):
    t0 = time.perf_counter()
    sim = make_simulation(seed)
    peaks = extract_peaks_once(sim)
    traj_cost = build_traj_cost(RECOMMENDED_TRAJ_SPEC)
    rows = []

    print("  Running all-distance tracking...")
    all_distance = run_all_particles_method(
        peaks,
        sim.trajectories_GT,
        build_distance_peak_cost(),
        traj_cost,
    )
    rows.append(mot_tracking_quality_row("all_distance", all_distance, sim.trajectories_GT, seed))

    print("  Running all-enhanced tracking...")
    all_enhanced = run_all_particles_method(
        peaks,
        sim.trajectories_GT,
        build_peak_cost(RECOMMENDED_PEAK_SPEC),
        traj_cost,
    )
    rows.append(mot_tracking_quality_row("all_enhanced", all_enhanced, sim.trajectories_GT, seed))

    print("  Running separated-distance tracking...")
    separated = run_separated_distance_method(
        peaks,
        sim.trajectories_GT,
        traj_cost,
    )
    rows.append(mot_tracking_quality_row("separated_distance", separated, sim.trajectories_GT, seed))

    if verbose:
        print(f"seed={seed}: done in {time.perf_counter() - t0:.2f}s")

    return rows


def display_columns():
    return [
        "seed",
        "method",
        "population",
        "method_label",
        "hota",
        "deta",
        "assa",
        "idf1",
        "id_precision",
        "id_recall",
        "mota",
        "motp_px",
        "id_switches",
        "link_precision",
        "link_recall",
        "link_f1",
        "coverage",
        "coverage_pct",
        "median_error_px",
        "mean_error_px",
        "fragmentations",
        "mean_fragmentations_per_gt",
        "n_tracks",
    ]


def summarize_results(results_df):
    rows = []
    metric_cols = [
        "hota",
        "deta",
        "assa",
        "idf1",
        "mota",
        "motp_px",
        "link_f1",
        "coverage",
        "coverage_pct",
        "median_error_px",
        "mean_error_px",
        "fragmentations",
        "mean_fragmentations_per_gt",
        "n_tracks",
    ]

    for method in METHOD_ORDER:
        group = results_df[results_df["method"] == method]
        if len(group) == 0:
            continue

        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "population": "all",
            "n": len(group),
        }

        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = finite_mean(values)
            row[f"{metric}_ci95"] = 1.96 * finite_sem(values)

        rows.append(row)

    return pd.DataFrame(rows)


def run_experiment(seeds=SEEDS, force_rerun=True, verbose=True):
    if RESULTS_PATH.exists() and not force_rerun:
        results_df = pd.read_csv(RESULTS_PATH)
        missing_columns = set(display_columns()) - set(results_df.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Cached AllvsCostvsSep HOTA results use an older schema. "
                f"Missing columns: {missing_text}. Rerun with force_rerun=True."
            )
        summary_df = summarize_results(results_df)
        return results_df, summary_df

    rows = []
    for seed in seeds:
        print(f"Running homogeneous seed {seed}/{seeds[-1]}")
        rows.extend(run_one_seed(seed, verbose=verbose))

    results_df = pd.DataFrame(rows, columns=display_columns())
    summary_df = summarize_results(results_df)

    results_df.to_csv(RESULTS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    return results_df, summary_df


# =============================================================================
# Ligand-receptor pipeline
# =============================================================================

def make_ligand_receptor_simulation(seed):
    return LigandReceptorSimulator(
        simulation_config=copy.deepcopy(LR_SIMULATION_CONFIG),
        image_config=copy.deepcopy(LR_IMAGE_CONFIG),
        seed=seed,
    ).run()


def extract_lr_peaks_once(sim):
    return extract_peaks(
        sim.frames,
        mode=LR_TRACKING_KWARGS["mode"],
        detection_threshold=LR_TRACKING_KWARGS["detection_threshold"],
        r_squared_threshold=LR_TRACKING_KWARGS["r_squared_threshold"],
        verbose_loc=False,
        visualization_loc=False,
    )


def lr_mobile_tracking_kwargs(peak_cost, traj_cost):
    return {
        "max_distance": LR_TRACKING_KWARGS["max_distance"],
        "min_length": LR_TRACKING_KWARGS["min_length"],
        "max_gap": LR_TRACKING_KWARGS["max_gap"],
        "stitch_fragments": LR_TRACKING_KWARGS["stitch_fragments"],
        "stitch_max_gap": LR_TRACKING_KWARGS["stitch_max_gap"],
        "stitch_base_distance": LR_TRACKING_KWARGS["stitch_base_distance"],
        "stitch_kwargs": LR_TRACKING_KWARGS["stitch_kwargs"],
        "algo_peak2peak": LR_TRACKING_KWARGS["algo_peak2peak"],
        "cost_func_peak2peak": peak_cost,
        "algo_traj2traj": LR_TRACKING_KWARGS["algo_traj2traj"],
        "cost_func_traj2traj": traj_cost,
        "verbose_assignment": False,
    }


def run_lr_all_particles_method(peaks, trajectories_gt, peak_cost, traj_cost):
    trajectories, _, _, _ = track_from_peaks(
        copy.deepcopy(peaks),
        trajectories_gt,
        **lr_mobile_tracking_kwargs(peak_cost, traj_cost),
    )
    return trajectories


def run_lr_separated_distance_method(frames, peaks, trajectories_gt, traj_cost):
    static_peak_cost = build_distance_peak_cost(LR_STATIC_TRACKING_KWARGS["max_distance"])
    mobile_peak_cost = build_distance_peak_cost(LR_TRACKING_KWARGS["max_distance"])

    static_peaks = extract_peaks(
        frames,
        mode=LR_TRACKING_KWARGS["mode"],
        detection_threshold=LR_STATIC_TRACKING_KWARGS["detection_threshold"],
        r_squared_threshold=LR_STATIC_TRACKING_KWARGS["r_squared_threshold"],
        verbose_loc=False,
        visualization_loc=False,
    )

    static_trajectories, _, _, _ = track_from_peaks(
        static_peaks,
        trajectories_gt,
        max_distance=LR_STATIC_TRACKING_KWARGS["max_distance"],
        min_length=LR_STATIC_TRACKING_KWARGS["min_length"],
        max_gap=LR_STATIC_TRACKING_KWARGS["max_gap"],
        algo_peak2peak=LR_TRACKING_KWARGS["algo_peak2peak"],
        cost_func_peak2peak=static_peak_cost,
        algo_traj2traj=LR_TRACKING_KWARGS["algo_traj2traj"],
        cost_func_traj2traj=traj_cost,
        verbose_assignment=False,
    )

    mobile_peaks = remove_static_peaks(
        copy.deepcopy(peaks),
        static_trajectories,
        tolerance=LR_STATIC_TRACKING_KWARGS["remove_tolerance"],
        verbose=False,
    )

    mobile_trajectories, _, _, _ = track_from_peaks(
        mobile_peaks,
        trajectories_gt,
        **lr_mobile_tracking_kwargs(mobile_peak_cost, traj_cost),
    )

    bridged_mobile_trajectories = bridge_mobile_fragments_through_static_anchors(
        mobile_trajectories,
        static_trajectories,
        **LR_BRIDGE_KWARGS,
    )

    combined = copy.deepcopy(bridged_mobile_trajectories) + copy.deepcopy(static_trajectories)

    if combined:
        combined, _, _ = assign_trajectories(
            combined,
            trajectories_gt,
            algorithm=LR_TRACKING_KWARGS["algo_traj2traj"],
            cost_function=traj_cost,
            verbose=False,
        )

    return combined


def lr_population_quality_rows(method_name, trajectories, trajectories_gt, seed, binding_events):
    rows = []

    for population in ["ligand", "receptor"]:
        gt_subset = filter_trajectories_by_population(trajectories_gt, population)

        # Keep all predicted trajectories here: HOTA matching decides which ones match the GT subset.
        # This penalizes cross-population false positives when they occur.
        row = mot_tracking_quality_row(
            method_name,
            trajectories,
            gt_subset,
            seed,
            population=population,
        )
        row["binding_events"] = int(len(binding_events))
        rows.append(row)

    return rows


def run_ligand_receptor_one_seed(seed, verbose=True):
    t0 = time.perf_counter()
    sim = make_ligand_receptor_simulation(seed)
    peaks = extract_lr_peaks_once(sim)
    traj_cost = build_traj_cost(RECOMMENDED_TRAJ_SPEC)
    rows = []

    all_distance = run_lr_all_particles_method(
        peaks,
        sim.trajectories_GT,
        build_distance_peak_cost(LR_TRACKING_KWARGS["max_distance"]),
        traj_cost,
    )
    rows.extend(
        lr_population_quality_rows(
            "all_distance",
            all_distance,
            sim.trajectories_GT,
            seed,
            sim.binding_events,
        )
    )

    all_enhanced = run_lr_all_particles_method(
        peaks,
        sim.trajectories_GT,
        build_peak_cost(RECOMMENDED_PEAK_SPEC, LR_PEAK_NORMS),
        traj_cost,
    )
    rows.extend(
        lr_population_quality_rows(
            "all_enhanced",
            all_enhanced,
            sim.trajectories_GT,
            seed,
            sim.binding_events,
        )
    )

    separated = run_lr_separated_distance_method(
        sim.frames,
        peaks,
        sim.trajectories_GT,
        traj_cost,
    )
    rows.extend(
        lr_population_quality_rows(
            "separated_distance",
            separated,
            sim.trajectories_GT,
            seed,
            sim.binding_events,
        )
    )

    if verbose:
        print(
            f"LR seed={seed}: done in {time.perf_counter() - t0:.2f}s, "
            f"binding events={len(sim.binding_events)}"
        )

    return rows


def lr_display_columns():
    return display_columns() + ["binding_events"]


def summarize_ligand_receptor_results(results_df):
    rows = []
    metric_cols = [
        "hota",
        "deta",
        "assa",
        "idf1",
        "mota",
        "motp_px",
        "link_f1",
        "coverage",
        "coverage_pct",
        "median_error_px",
        "mean_error_px",
        "fragmentations",
        "mean_fragmentations_per_gt",
        "n_tracks",
        "binding_events",
    ]

    for population in ["ligand", "receptor"]:
        for method in METHOD_ORDER:
            group = results_df[
                (results_df["population"] == population)
                & (results_df["method"] == method)
            ]
            if len(group) == 0:
                continue

            row = {
                "population": population,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n": len(group),
            }

            for metric in metric_cols:
                values = group[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = finite_mean(values)
                row[f"{metric}_ci95"] = 1.96 * finite_sem(values)

            rows.append(row)

    return pd.DataFrame(rows)


def run_ligand_receptor_experiment(seeds=LR_SEEDS, force_rerun=True, verbose=True):
    if LR_RESULTS_PATH.exists() and not force_rerun:
        results_df = pd.read_csv(LR_RESULTS_PATH)
        missing_columns = set(lr_display_columns()) - set(results_df.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Cached ligand-receptor HOTA results use an older schema. "
                f"Missing columns: {missing_text}. Rerun with force_rerun=True."
            )
        summary_df = summarize_ligand_receptor_results(results_df)
        return results_df, summary_df

    rows = []
    for seed in seeds:
        print(f"Running ligand-receptor seed {seed}/{seeds[-1]}")
        rows.extend(run_ligand_receptor_one_seed(seed, verbose=verbose))

    results_df = pd.DataFrame(rows, columns=lr_display_columns())
    summary_df = summarize_ligand_receptor_results(results_df)

    results_df.to_csv(LR_RESULTS_PATH, index=False)
    summary_df.to_csv(LR_SUMMARY_PATH, index=False)

    return results_df, summary_df


# =============================================================================
# Plotting helpers
# =============================================================================

def add_panel_label(ax, label, x=0.02, y=0.95):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _method_colors(n):
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    return colors[:n]


def plot_mot_summary(summary_df, save_path=FIGURE_PATH, title="Tracking strategy comparison"):
    methods = summary_df["method_label"].tolist()
    x = np.arange(len(summary_df))
    colors = _method_colors(len(summary_df))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    # Panel A: HOTA, DetA, AssA
    width = 0.25
    axes[0].bar(x - width, summary_df["hota_mean"], width, label="HOTA", color="#2563eb")
    axes[0].bar(x, summary_df["deta_mean"], width, label="DetA", color="#16a34a")
    axes[0].bar(x + width, summary_df["assa_mean"], width, label="AssA", color="#dc2626")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("MOT accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel B: coverage-error trade-off
    error = summary_df["median_error_px_mean"].to_numpy(dtype=float)
    coverage = summary_df["coverage_mean"].to_numpy(dtype=float)
    n_tracks = summary_df["n_tracks_mean"].to_numpy(dtype=float)

    if np.nanmax(n_tracks) > np.nanmin(n_tracks):
        sizes = 180 + 850 * (n_tracks - np.nanmin(n_tracks)) / (np.nanmax(n_tracks) - np.nanmin(n_tracks))
    else:
        sizes = np.full_like(n_tracks, 500.0)

    for i, label in enumerate(methods):
        axes[1].scatter(
            error[i],
            coverage[i],
            s=sizes[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            label=f"{label} (n={n_tracks[i]:.1f})",
        )
        axes[1].annotate(label, (error[i], coverage[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    axes[1].set_xlabel("Median assigned-position error (px)")
    axes[1].set_ylabel("Mean GT coverage")
    axes[1].set_title("Coverage--error trade-off")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    # Panel C: IDF1, MOTA, MOTP
    axes[2].bar(x - width, summary_df["idf1_mean"], width, label="IDF1", color="#2563eb")
    axes[2].bar(x, summary_df["mota_mean"], width, label="MOTA", color="#16a34a")

    motp = summary_df["motp_px_mean"].to_numpy(dtype=float)
    if np.nanmax(motp) > 0:
        motp_score = 1.0 - motp / np.nanmax(motp)
    else:
        motp_score = np.ones_like(motp)

    axes[2].bar(x + width, motp_score, width, label="1 - normalized MOTP", color="#dc2626")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=20, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Identity and precision diagnostics")
    axes[2].set_ylabel("Score")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    for label, ax in zip("ABC", axes):
        add_panel_label(ax, label)

    fig.suptitle(title, y=1.04)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_ligand_receptor_mot_summary(summary_df, save_path=LR_FIGURE_PATH):
    populations = ["ligand", "receptor"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    panel_labels = list("ABCDEF")
    label_idx = 0

    for row_idx, population in enumerate(populations):
        sub = summary_df[summary_df["population"] == population].copy()
        methods = sub["method_label"].tolist()
        x = np.arange(len(sub))
        colors = _method_colors(len(sub))

        # HOTA decomposition
        width = 0.25
        ax = axes[row_idx, 0]
        ax.bar(x - width, sub["hota_mean"], width, label="HOTA", color="#2563eb")
        ax.bar(x, sub["deta_mean"], width, label="DetA", color="#16a34a")
        ax.bar(x + width, sub["assa_mean"], width, label="AssA", color="#dc2626")
        ax.set_title(f"{population.capitalize()} MOT accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        add_panel_label(ax, panel_labels[label_idx])
        label_idx += 1

        # Coverage-error scatter
        ax = axes[row_idx, 1]
        error = sub["median_error_px_mean"].to_numpy(dtype=float)
        coverage = sub["coverage_mean"].to_numpy(dtype=float)
        n_tracks = sub["n_tracks_mean"].to_numpy(dtype=float)

        if np.nanmax(n_tracks) > np.nanmin(n_tracks):
            sizes = 180 + 850 * (n_tracks - np.nanmin(n_tracks)) / (np.nanmax(n_tracks) - np.nanmin(n_tracks))
        else:
            sizes = np.full_like(n_tracks, 500.0)

        for i, label in enumerate(methods):
            ax.scatter(
                error[i],
                coverage[i],
                s=sizes[i],
                color=colors[i],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                label=f"{label} (n={n_tracks[i]:.1f})",
            )
            ax.annotate(label, (error[i], coverage[i]), xytext=(5, 5), textcoords="offset points", fontsize=7)

        ax.set_xlabel("Median error (px)")
        ax.set_ylabel("Mean GT coverage")
        ax.set_title(f"{population.capitalize()} coverage--error")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6)
        add_panel_label(ax, panel_labels[label_idx])
        label_idx += 1

        # IDF1/MOTA/MOTP diagnostics
        ax = axes[row_idx, 2]
        ax.bar(x - width, sub["idf1_mean"], width, label="IDF1", color="#2563eb")
        ax.bar(x, sub["mota_mean"], width, label="MOTA", color="#16a34a")

        motp = sub["motp_px_mean"].to_numpy(dtype=float)
        if np.nanmax(motp) > 0:
            motp_score = 1.0 - motp / np.nanmax(motp)
        else:
            motp_score = np.ones_like(motp)

        ax.bar(x + width, motp_score, width, label="1 - normalized MOTP", color="#dc2626")
        ax.set_title(f"{population.capitalize()} identity diagnostics")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        add_panel_label(ax, panel_labels[label_idx])
        label_idx += 1

    fig.suptitle("Ligand--receptor tracking strategy comparison using MOT metrics", y=1.03)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# Convenience tables
# =============================================================================

def configuration_table():
    rows = [{"group": "simulation", "parameter": key, "value": value} for key, value in SIMULATION_CONFIG.items()]
    rows += [{"group": "image", "parameter": key, "value": value} for key, value in IMAGE_CONFIG.items()]
    rows += [{"group": "tracking", "parameter": key, "value": value} for key, value in TRACKING_KWARGS.items()]
    rows += [{"group": "static pass", "parameter": key, "value": value} for key, value in STATIC_TRACKING_KWARGS.items()]
    return pd.DataFrame(rows)


def ligand_receptor_configuration_table():
    rows = [{"group": "simulation", "parameter": key, "value": value} for key, value in LR_SIMULATION_CONFIG.items()]
    rows += [{"group": "image", "parameter": key, "value": value} for key, value in LR_IMAGE_CONFIG.items()]
    rows += [{"group": "tracking", "parameter": key, "value": value} for key, value in LR_TRACKING_KWARGS.items()]
    rows += [{"group": "static pass", "parameter": key, "value": value} for key, value in LR_STATIC_TRACKING_KWARGS.items()]
    return pd.DataFrame(rows)


def cost_table():
    return pd.DataFrame(
        [
            {
                "role": "all/separated distance peak cost",
                "name": "distance_only",
                "distance": 1.0,
                "intensity": 0.0,
                "sigma": 0.0,
                "position": np.nan,
                "length": np.nan,
                "start_frame": np.nan,
            },
            {
                "role": "all enhanced peak cost",
                **RECOMMENDED_PEAK_SPEC,
                "position": np.nan,
                "length": np.nan,
                "start_frame": np.nan,
            },
            {
                "role": "shared GT assignment/evaluation cost",
                "distance": np.nan,
                "intensity": np.nan,
                "sigma": np.nan,
                **RECOMMENDED_TRAJ_SPEC,
            },
        ]
    )


if __name__ == "__main__":
    df, summary = run_experiment(force_rerun=True, verbose=True)
    print(df[display_columns()].round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))
    plot_mot_summary(summary, save_path=FIGURE_PATH)
