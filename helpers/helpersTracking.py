# =============================================================================
# Imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import torch
import torch.nn as nn

from helpers.helpersGeneration import Trajectory
from helpers.helpersAssignment import *
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit

# =============================================================================
# Peak-to-peak cost functions
# =============================================================================

class CostTerm(nn.Module):
    def __init__(self, weight=1.0, norm=1.0, enabled=True):
        super().__init__()
        self.weight = weight
        self.norm = norm
        self.enabled = enabled

    def forward(self, trajectory, detection):
        raise NotImplementedError


class DistanceTerm(CostTerm):
    def forward(self, trajectory, detection):
        if not self.enabled:
            return torch.tensor(0.0)

        pos, _, _ = detection

        last_pos = torch.as_tensor(
            trajectory.last_position(),
            dtype=torch.float32,
        )

        pos = torch.as_tensor(pos, dtype=torch.float32)

        dist = torch.linalg.norm(last_pos - pos)
        return self.weight * (dist / self.norm)


class IntensityTerm(CostTerm):
    def forward(self, trajectory, detection):
        if not self.enabled:
            return torch.tensor(0.0)

        _, det_intensity, _ = detection
        last_intensity = trajectory.last_intensity()

        if last_intensity is None or det_intensity is None:
            return torch.tensor(0.0)

        last_intensity = torch.as_tensor(last_intensity, dtype=torch.float32)
        det_intensity = torch.as_tensor(det_intensity, dtype=torch.float32)

        diff = torch.abs(last_intensity - det_intensity)
        return self.weight * (diff / self.norm)


class SigmaTerm(CostTerm):
    def forward(self, trajectory, detection):
        if not self.enabled:
            return torch.tensor(0.0)

        _, _, det_sigma = detection # ((x,y),int,sigma)
        last_sigma = trajectory.last_sigma()

        if last_sigma is None or det_sigma is None:
            return torch.tensor(0.0)

        last_sigma = torch.as_tensor(last_sigma, dtype=torch.float32)
        det_sigma = torch.as_tensor(det_sigma, dtype=torch.float32)

        diff = torch.abs(last_sigma - det_sigma)
        return self.weight * (diff / self.norm)


class PeakToPeak(nn.Module):
    def __init__(self, terms=None, return_breakdown=False):
        super().__init__()

        if terms is None:
            terms = {}

        self.terms = nn.ModuleDict(terms)
        self.return_breakdown = return_breakdown

    @classmethod
    def default(cls):
        return cls(
            terms={
                "distance": DistanceTerm(weight=0.5, norm=0.25 * 128),
                "intensity": IntensityTerm(weight=0.4, norm=650),
                "sigma": SigmaTerm(weight=0.1, norm=2.0),
            }
        )

    def forward(self, trajectory, detection):
        costs = {
            name: term(trajectory, detection)
            for name, term in self.terms.items()
        }

        total = sum(costs.values())

        if self.return_breakdown:
            return total, costs

        return total
    
# =============================================================================
# Peak detection
# =============================================================================

def detect_peaks(frames, threshold_abs=500, min_distance=1):    
    detected_peaks = []
    for frame in frames:
        peaks = peak_local_max(frame, min_distance=min_distance, threshold_abs=threshold_abs)
        detected_peaks.append(peaks)
    return detected_peaks # returns (row, col)

# visualize quality of peak detection
def visualize_peaks(peaks, bg_frame, frame_index):
    plt.imshow(bg_frame, cmap='gray')
    plt.scatter(peaks[:, 1], peaks[:, 0], c='r', marker='x')
    plt.title(f'Detected peaks - frame {frame_index+1}')
    plt.show()

# =============================================================================
# Gaussian localization
# =============================================================================

def amp(sigma, A):
    """Convert integrated Gaussian intensity to peak amplitude."""
    return A / (2 * np.pi * sigma**2)


def gaussian_2d(xy, x0, y0, A, sigma, B):
    """
    Evaluate a 2D isotropic Gaussian with constant background.

    Parameters
    ----------
    xy : tuple of ndarray
        Meshgrid coordinates ``(x, y)``.
    x0, y0 : float
        Gaussian center coordinates in patch coordinates.
    A : float
        Integrated Gaussian intensity.
    sigma : float
        Gaussian standard deviation.
    B : float
        Constant background offset.

    Returns
    -------
    ndarray
        Gaussian intensity evaluated at each coordinate.
    """
    x, y = xy

    return B + amp(sigma, A) * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)
    )


def fit_gaussian_to_peak(frame, peak, verbose=False):
    """
    Fit a 2D Gaussian to a 5x5 patch centered on a detected peak.

    Parameters
    ----------
    frame : ndarray
        Image frame containing the detected peak.
    peak : tuple
        Integer peak coordinates as ``(row, col)``.
    verbose : bool, optional
        Currently unused. Kept for API compatibility.

    Returns
    -------
    tuple or None
        Returns ``(row_fit, col_fit, A, sigma, B, r_squared)`` if the fit
        succeeds. Returns ``None`` if the patch is incomplete or the fit fails.
    """
    row, col = peak
    patch = frame[row - 2:row + 3, col - 2:col + 3]

    if patch.shape != (5, 5):
        return None

    patch = np.asarray(patch, dtype=float)

    x_data = np.arange(patch.shape[1])
    y_data = np.arange(patch.shape[0])
    x_data, y_data = np.meshgrid(x_data, y_data)

    sigma0 = 0.5
    peak_height0 = np.max(patch)
    A0 = 1000 * sigma0**2 * 2 * np.pi
    B0 = 100

    initial_guess = (2.0, 2.0, A0, sigma0, B0)

    bounds = (
        [0.0, 0.0, 0.0, 0.3, 0.0],
        [4.0, 4.0, np.inf, 4.0, np.inf],
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (x_data.ravel(), y_data.ravel()),
            patch.ravel(),
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    x0_fit, y0_fit, A_fit, sigma_fit, B_fit = popt

    row_fit_img = (row - 2) + y0_fit
    col_fit_img = (col - 2) + x0_fit

    yfit = gaussian_2d((x_data, y_data), *popt).reshape(patch.shape)
    residuals = patch - yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((patch - np.mean(patch)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return row_fit_img, col_fit_img, A_fit, sigma_fit, B_fit, r_squared


def localize_peaks_with_gaussian_fitting(
    frames,
    detected_peaks,
    r_squared_threshold=0.5,
    verbose=False,
    visualization=False,
    visualization_peak_idx=0,
):
    """
    Localize detected peaks by fitting a 2D Gaussian to each peak neighborhood.

    A 5x5 patch is extracted around each detected peak. The fitted Gaussian
    center provides subpixel localization, while the fitted amplitude and sigma
    are retained for downstream tracking costs.

    Parameters
    ----------
    frames : sequence of ndarray
        Image frames.
    detected_peaks : sequence of sequence
        Detected peak coordinates for each frame, typically returned by
        ``detect_peaks``.
    r_squared_threshold : float, optional
        Minimum coefficient of determination required to keep a fitted peak.
    verbose : bool, optional
        Currently unused. Kept for API compatibility.
    visualization : bool, optional
        If True, visualize one selected Gaussian fit.
    visualization_peak_idx : int, optional
        Peak index to visualize in the first frame.

    Returns
    -------
    list of list of tuple
        Localized detections for each frame. Each detection is stored as
        ``((row, col), amplitude, sigma, r_squared)``.
    """
    localized_peaks = []

    for frame_idx, (frame, peaks) in enumerate(zip(frames, detected_peaks)):
        localized_frame_peaks = []

        for peak_idx, peak in enumerate(peaks):
            fitted = fit_gaussian_to_peak(frame, peak, verbose=verbose)

            pos_fit = (fitted[0], fitted[1]) if fitted is not None else None
            A_fit = fitted[2] if fitted is not None else None
            sigma_fit = fitted[3] if fitted is not None else None
            r_squared = fitted[5] if fitted is not None else None

            amp_fit = (
                A_fit / (2 * np.pi * sigma_fit**2)
                if A_fit is not None and sigma_fit is not None
                else None
            )

            localized_peak = (
                (pos_fit, amp_fit, sigma_fit, r_squared)
                if (
                    pos_fit is not None
                    and amp_fit is not None
                    and sigma_fit is not None
                )
                else None
            )

            if (
                localized_peak is not None
                and r_squared is not None
                and r_squared >= r_squared_threshold
            ):
                localized_frame_peaks.append(localized_peak)
                # print(
                #     f"[Frame {frame_idx}] Fitted center for peak at "
                #     f"({peak[0]}, {peak[1]}): "
                #     f"({pos_fit[0]:.2f}, {pos_fit[1]:.2f}), "
                #     f"amplitude {amp_fit:.2f}, sigma {sigma_fit:.2f}"
                # )

            elif r_squared is None:
                print(
                    f"[Frame {frame_idx}] No computed R-squared for peak at "
                    f"({peak[0]}, {peak[1]})."
                )

            elif r_squared < r_squared_threshold:
                print(
                    f"[Frame {frame_idx}] Poor fit for peak at "
                    f"({peak[0]}, {peak[1]}): R-squared = {r_squared:.3f} "
                    f"(below threshold of {r_squared_threshold})."
                )

            if (
                visualization
                and frame_idx == 0
                and peak_idx == visualization_peak_idx
                and fitted is not None
            ):
                visualize_gaussian_fit(frame, peak, fitted)

        localized_peaks.append(localized_frame_peaks)

    return localized_peaks


def visualize_gaussian_fit(frame, peak, fitted_params):
    """
    Visualize the Gaussian fit for a detected peak.

    Displays the local patch around the detected peak, the original detected
    peak position, the fitted subpixel center, and a circle with radius equal
    to the fitted Gaussian sigma.
    """
    row, col = peak
    row_fit, col_fit, A_fit, sigma_fit, B_fit, r_squared = fitted_params

    half = 2

    r0 = int(row)
    c0 = int(col)

    r_min = max(r0 - half, 0)
    r_max = min(r0 + half + 1, frame.shape[0])
    c_min = max(c0 - half, 0)
    c_max = min(c0 + half + 1, frame.shape[1])

    zoom = frame[r_min:r_max, c_min:c_max]

    plt.figure(figsize=(6, 6))

    plt.imshow(
        zoom,
        cmap="gray",
        origin="upper",
        extent=[0, zoom.shape[1], zoom.shape[0], 0],
    )

    peak_x = (col - c_min) + 0.5
    peak_y = (row - r_min) + 0.5

    plt.scatter(peak_x, peak_y, c="r", marker="x", label="Detected peak")

    fit_x = (col_fit - c_min) + 0.5
    fit_y = (row_fit - r_min) + 0.5

    plt.scatter(fit_x, fit_y, c="b", marker="o", label="Fitted center")

    radius = sigma_fit
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = fit_x + radius * np.cos(theta)
    circle_y = fit_y + radius * np.sin(theta)

    plt.plot(circle_x, circle_y, "b--", linewidth=1.5, label="σ radius")

    plt.xlim(0, zoom.shape[1])
    plt.ylim(zoom.shape[0], 0)

    plt.title(
        f"Gaussian fit ({half * 2 + 1}×{half * 2 + 1} zoom), "
        f"R²={r_squared:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# Frame-to-frame trajectory linking
# =============================================================================

def track_peaks_to_trajectories(
    peaks,
    max_distance=5,
    min_length=5,
    algorithm='hungarian',
    cost_function=None,
    max_gap=0,
):
    """
    General frame-to-frame tracking using detections of the form:
        ((x, y), intensity)
    or
        (x, y)

    ``max_gap`` keeps unmatched trajectories alive for a few frames so fast or
    dim particles can be reconnected after missed detections. The spatial gate
    grows as ``sqrt(number_of_elapsed_frames)`` across a gap.
    """
    if len(peaks) == 0:
        return [], 0.0

    trajectories = []
    active = []
    min_cost_frames = []

    for f, frame_peaks in enumerate(peaks):
        current_detections = []

        # normalize detection format
        for det in frame_peaks:
            pos = tuple(det[0])

            amp = None if det[1] is None else float(det[1])
            det_sigma = None if det[2] is None else float(det[2])

            current_detections.append((pos, amp, det_sigma))

        if len(current_detections) == 0:
            active = [
                idx for idx in active
                if f - trajectories[idx].end_frame <= max_gap
            ]
            continue

        active = [
            idx for idx in active
            if f - trajectories[idx].end_frame - 1 <= max_gap
        ]

        # initialize all detections if no active tracks
        if len(active) == 0:
            new_active = []
            for det in current_detections:
                pos, amp, det_sigma = det
                traj_id = len(trajectories)
                traj = Trajectory(traj_id, start_frame=f)
                traj.add_position(tuple(pos), frame=f, intensity=amp, sigma=det_sigma)
                trajectories.append(traj)
                new_active.append(traj_id)
            active = new_active
            continue

        active_trajectories = [trajectories[idx] for idx in active]
        if max_distance is None:
            row_max_distances = None
        else:
            frame_gaps = np.array(
                [max(f - trajectories[idx].end_frame, 1) for idx in active],
                dtype=float,
            )
            row_max_distances = max_distance * np.sqrt(frame_gaps)

        if cost_function is None:
            cost_function = PeakToPeak.default()

        cost_matrix = compute_cost_matrix_tracks_to_detections(
            active_trajectories,
            current_detections,
            cost_function=cost_function,
            max_distance=row_max_distances,
        )

        gated_cost = cost_matrix.copy()

        if algorithm == 'hungarian':
            min_cost_frame, assignment = hungarian(gated_cost)
        elif algorithm == 'local_nn':
            assignment_threshold = (
                np.inf if row_max_distances is None else np.max(row_max_distances)
            )
            min_cost_frame, assignment = local_nn_assignment(gated_cost, max_distance=assignment_threshold)
        elif algorithm == 'global_nn':
            assignment_threshold = (
                np.inf if row_max_distances is None else np.max(row_max_distances)
            )
            min_cost_frame, assignment = global_nn_assignment(gated_cost, max_distance=assignment_threshold)
        else:
            raise ValueError("Invalid algorithm. Choose 'hungarian', 'local_nn', or 'global_nn'.")
        
        min_cost_frames.append(min_cost_frame) #TODO: study shape of min_cost_frames...

        used = np.zeros(len(current_detections), dtype=bool)
        new_active = []
        matched_active = set()

        for row_idx, det_idx in enumerate(assignment):
            traj_idx = active[row_idx]

            if det_idx == -1:
                continue

            if cost_matrix[row_idx, det_idx] >= 1e6:
                continue

            pos, amp, sigma = current_detections[det_idx]

            trajectories[traj_idx].add_position(
                tuple(pos),
                frame=f,
                intensity=amp,
                sigma=sigma
            )
            used[det_idx] = True
            new_active.append(traj_idx)
            matched_active.add(traj_idx)

        for traj_idx in active:
            if traj_idx in matched_active:
                continue
            if f - trajectories[traj_idx].end_frame <= max_gap:
                new_active.append(traj_idx)

        for j, det in enumerate(current_detections):
            if not used[j]:
                pos, amp, sigma = det
                traj_id = len(trajectories)
                traj = Trajectory(traj_id, start_frame=f)
                traj.add_position(tuple(pos), frame=f, intensity=amp, sigma=sigma)
                trajectories.append(traj)
                new_active.append(traj_id)

        active = list(dict.fromkeys(new_active))

    trajectories = [traj for traj in trajectories if traj.length() >= min_length]
    return trajectories, min_cost_frames 


def _trajectory_first_intensity(trajectory):
    return trajectory.intensities[0] if trajectory.intensities else None


def _trajectory_first_sigma(trajectory):
    return trajectory.sigmas[0] if trajectory.sigmas else None


def _merge_trajectory_chain(chain, new_id):
    merged = copy.deepcopy(chain[0])
    merged.id = new_id

    for fragment in chain[1:]:
        for frame, position, intensity, sigma, state, bound_to in zip(
            fragment.frames(),
            fragment.positions,
            fragment.intensities,
            fragment.sigmas,
            fragment.states,
            fragment.bound_to,
        ):
            merged.add_position(
                position,
                frame=frame,
                intensity=intensity,
                sigma=sigma,
                state=state,
                bound_to=bound_to,
            )

    return merged


def stitch_trajectory_fragments(
    trajectories,
    max_gap=4,
    base_distance=10.0,
    distance_weight=1.0,
    gap_weight=0.1,
    intensity_weight=0.2,
    sigma_weight=0.2,
    intensity_scale=500.0,
    sigma_scale=0.75,
    use_intensity=True,
    use_sigma=True,
    max_link_cost=2.0,
    max_iterations=5,
    verbose=False,
):
    """
    Stitch temporally separated trajectory fragments before GT assignment.

    Fragments are linked only if they do not overlap in time and the gap is no
    larger than ``max_gap`` missed frames. The spatial gate grows as
    ``base_distance * sqrt(elapsed_frames)``, which is a simple Brownian-motion
    prior for fast ligands.
    """
    stitched = [copy.deepcopy(traj) for traj in trajectories if traj.length() > 0]

    if len(stitched) <= 1:
        return stitched

    for iteration in range(max_iterations):
        n = len(stitched)
        invalid_cost = 1e6
        cost_matrix = np.full((n, n), invalid_cost, dtype=float)

        for i, tail in enumerate(stitched):
            tail_pos = np.asarray(tail.last_position(), dtype=float)
            tail_intensity = tail.last_intensity()
            tail_sigma = tail.last_sigma()

            for j, head in enumerate(stitched):
                if i == j:
                    continue

                elapsed_frames = head.start_frame - tail.end_frame
                missing_frames = elapsed_frames - 1

                if elapsed_frames <= 0 or missing_frames > max_gap:
                    continue

                head_pos = np.asarray(head.positions[0], dtype=float)
                allowed_distance = base_distance * np.sqrt(elapsed_frames)
                distance = np.linalg.norm(head_pos - tail_pos)

                if distance > allowed_distance:
                    continue

                cost = distance_weight * (distance / allowed_distance)
                cost += gap_weight * missing_frames

                if use_intensity:
                    head_intensity = _trajectory_first_intensity(head)
                    if tail_intensity is not None and head_intensity is not None:
                        cost += intensity_weight * (
                            abs(tail_intensity - head_intensity) / intensity_scale
                        )

                if use_sigma:
                    head_sigma = _trajectory_first_sigma(head)
                    if tail_sigma is not None and head_sigma is not None:
                        cost += sigma_weight * (
                            abs(tail_sigma - head_sigma) / sigma_scale
                        )

                if cost <= max_link_cost:
                    cost_matrix[i, j] = cost

        if not np.any(cost_matrix < invalid_cost):
            break

        _, assignment = hungarian(cost_matrix)
        successor = {}
        predecessors = set()

        for i, j in enumerate(assignment):
            if j == -1 or cost_matrix[i, j] >= invalid_cost:
                continue
            successor[i] = int(j)
            predecessors.add(int(j))

        if len(successor) == 0:
            break

        merged = []
        visited = set()
        starts = [i for i in range(n) if i not in predecessors]

        for start in starts:
            if start in visited:
                continue

            chain_indices = []
            current = start

            while current not in visited:
                visited.add(current)
                chain_indices.append(current)
                if current not in successor:
                    break
                current = successor[current]

            chain = [stitched[idx] for idx in chain_indices]
            merged.append(_merge_trajectory_chain(chain, new_id=len(merged)))

        for idx in range(n):
            if idx not in visited:
                orphan = copy.deepcopy(stitched[idx])
                orphan.id = len(merged)
                merged.append(orphan)

        if verbose:
            print(
                f"Stitch iteration {iteration + 1}: "
                f"{n} -> {len(merged)} trajectories "
                f"({len(successor)} links)"
            )

        if len(merged) == n:
            break

        stitched = merged

    for idx, traj in enumerate(stitched):
        traj.id = idx

    return stitched

# =============================================================================
# Full tracking pipeline with detection or detection+localization
# =============================================================================

def remove_static_peaks(peaks_for_tracking,
                        static_trajectories,
                        tolerance=5.0,
                        verbose=False):
    """
    Remove peaks that correspond to static trajectories by matching each
    static trajectory position to the closest localized peak in the same frame.

    Parameters
    ----------
    peaks_for_tracking : list[list]
        peaks_for_tracking[frame] = list of peaks, each peak is (pos, intensity, ...)
    static_trajectories : list
        List of trajectory objects with attributes:
            - id
            - start_frame
            - end_frame
            - positions (list of (x, y))
    tolerance : float
        Maximum allowed distance to consider a peak as belonging to a static trajectory.
    verbose : bool
        If True, prints detailed removal logs.

    Returns
    -------
    cleaned_peaks : list[list]
        Deep copy of peaks_for_tracking with static peaks removed.
    """

    cleaned_peaks = copy.deepcopy(peaks_for_tracking)

    for traj in static_trajectories:
        removed_count = 0

        for frame_id, position in zip(traj.frames(), traj.positions):
            traj_pos = np.array(position)

            peak_list = cleaned_peaks[frame_id]
            if not peak_list:
                continue

            # Find closest peak
            closest_idx = None
            closest_dist = float("inf")

            for idx, peak in enumerate(peak_list):
                peak_pos = np.array(peak[0])
                dist = np.linalg.norm(peak_pos - traj_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = idx

            # Remove if within tolerance
            if closest_dist < tolerance:
                removed_peak = peak_list.pop(closest_idx)
                removed_count += 1

                # if verbose:
                #     print(
                #         f"[Frame {frame_id}] Removed peak {removed_peak[0]} "
                #         f"closest to static position {traj_pos.tolist()} "
                #         f"(dist={closest_dist:.2f})"
                #     )

        if verbose:
            length = traj.length()
            start = traj.start_frame
            end = traj.end_frame
            print(
                f"Removed {removed_count} peaks from static trajectory {traj.id} "
                f"(length {length}, frames {start}–{end})"
            )

    return cleaned_peaks

def track(
    frames,
    trajectories_GT,
    mode="localization",
    detection_threshold=None,
    max_distance=10,
    min_length=5,
    max_gap=0,
    stitch_fragments=False,
    stitch_max_gap=4,
    stitch_base_distance=10.0,
    stitch_kwargs=None,
    verbose_stitching=False,
    r_squared_threshold=0.5,
    algo_peak2peak="hungarian",
    cost_func_peak2peak=None,
    algo_traj2traj="hungarian",
    cost_func_traj2traj=None,
    verbose_loc=False,
    visualization_loc=False,
    verbose_assignment=False,
):
    """
    Run the full tracking pipeline.

    The pipeline performs peak detection, subpixel Gaussian localization,
    frame-to-frame trajectory linking, and assignment of localized trajectories
    to ground-truth trajectories.

    Parameters
    ----------
    frames : sequence of ndarray
        Image frames to process.
    trajectories_GT : list of Trajectory
        Ground-truth trajectories used for final trajectory assignment.
    max_distance : float, optional
        Maximum allowed distance for frame-to-frame linking.
    min_length : int, optional
        Minimum trajectory length retained after peak-to-peak tracking.
    max_gap : int, optional
        Number of missed frames allowed before a track is terminated.
    stitch_fragments : bool, optional
        If True, stitch compatible trajectory fragments before GT assignment.
    r_squared_threshold : float, optional
        Minimum Gaussian fit quality required to keep a localized peak.
    algo_peak2peak : {"hungarian", "local_nn", "global_nn"}, optional
        Assignment algorithm used for frame-to-frame peak linking.
    algo_traj2traj : {"hungarian", "local_nn", "global_nn"}, optional
        Assignment algorithm used to match localized trajectories to
        ground-truth trajectories.
    cost_function : callable, optional
        Custom peak-to-peak cost function. If None, the default cost is used.
    verbose_loc : bool, optional
        If True, print localization details.
    visualization_loc : bool, optional
        If True, visualize one Gaussian localization fit.
    verbose_assignment : bool, optional
        If True, print trajectory assignment details.

    Returns
    -------
    tuple
        ``(trajectories_localization, cost_peak2peak,
        assignment_localization, cost_traj2traj)``.
    """ 
    if detection_threshold is None:
        max = np.max(frames)
        min = np.min(frames)
        k = 0.7
        detection_threshold = max - k*(max - min)

    detected_peaks = detect_peaks(
    frames,
    threshold_abs=detection_threshold,
    min_distance=1,
    )

    if mode == "detection":
        peaks_for_tracking = []

        for frame, peaks in zip(frames, detected_peaks):
            frame_detections = []

            for row, col in peaks:
                pos = (float(row), float(col))
                intensity = float(frame[row, col])
                sigma = None

                frame_detections.append((pos, intensity, sigma))

            peaks_for_tracking.append(frame_detections)

    elif mode == "localization":
        peaks_for_tracking = localize_peaks_with_gaussian_fitting(
            frames,
            detected_peaks,
            r_squared_threshold=r_squared_threshold,
            verbose=verbose_loc,
            visualization=visualization_loc,
            visualization_peak_idx=1,
        )

    else:
        raise ValueError("mode must be 'detection' or 'localization'.")

    trajectories_output, cost_peak2peak = track_peaks_to_trajectories(
        peaks=peaks_for_tracking,
        max_distance=max_distance,
        min_length=min_length,
        algorithm=algo_peak2peak,
        cost_function=cost_func_peak2peak,
        max_gap=max_gap,
    )

    if stitch_fragments and len(trajectories_output) > 0:
        stitch_kwargs = stitch_kwargs or {}
        trajectories_output = stitch_trajectory_fragments(
            trajectories_output,
            max_gap=stitch_max_gap,
            base_distance=stitch_base_distance,
            verbose=verbose_stitching,
            **stitch_kwargs,
        )

    if len(trajectories_output) == 0:
        return [], cost_peak2peak, [], np.inf

    (
        trajectories_output,
        cost_traj2traj,
        assignment_output,
    ) = assign_trajectories(
        trajectories_output,
        trajectories_GT,
        algorithm=algo_traj2traj,
        cost_function=cost_func_traj2traj,
        verbose=verbose_assignment,
    )

    return (
        trajectories_output,
        cost_peak2peak,
        assignment_output,
        cost_traj2traj,
    )

def extract_peaks(
    frames, 
    mode='detection', 
    detection_threshold=None, 
    r_squared_threshold=0.5, 
    verbose_loc=False, 
    visualization_loc=False):
    """Performs peak detection and optional Gaussian localization on the input frames."""
    
    if detection_threshold is None:
        max = np.max(frames)
        min = np.min(frames)
        k = 0.7
        detection_threshold = max - k*(max - min)
        
    detected_peaks = detect_peaks(
    frames,
    threshold_abs=detection_threshold,
    min_distance=1,
    )

    if mode == "detection":
        extracted_peaks = []

        for frame, peaks in zip(frames, detected_peaks):
            frame_detections = []

            for row, col in peaks:
                pos = (float(row), float(col))
                intensity = float(frame[row, col])
                sigma = None

                frame_detections.append((pos, intensity, sigma))

            extracted_peaks.append(frame_detections)

    elif mode == "localization":
        extracted_peaks = localize_peaks_with_gaussian_fitting(
            frames,
            detected_peaks,
            r_squared_threshold=r_squared_threshold,
            verbose=verbose_loc,
            visualization=visualization_loc,
            visualization_peak_idx=1,
        )

    else:
        raise ValueError("mode must be 'detection' or 'localization'.")
    
    return extracted_peaks

def track_from_peaks(
    peaks_for_tracking, 
    trajectories_GT, 
    max_distance=10, 
    min_length=5, 
    max_gap=0,
    stitch_fragments=False,
    stitch_max_gap=4,
    stitch_base_distance=10.0,
    stitch_kwargs=None,
    verbose_stitching=False,
    algo_peak2peak="hungarian", 
    cost_func_peak2peak=None, 
    algo_traj2traj="hungarian", 
    cost_func_traj2traj=None, 
    verbose_assignment=False):
    """Performs frame-to-frame tracking and trajectory assignment starting from pre-extracted peaks."""
    trajectories_output, cost_peak2peak = track_peaks_to_trajectories(
        peaks=peaks_for_tracking,
        max_distance=max_distance,
        min_length=min_length,
        algorithm=algo_peak2peak,
        cost_function=cost_func_peak2peak,
        max_gap=max_gap,
    )

    if stitch_fragments and len(trajectories_output) > 0:
        stitch_kwargs = stitch_kwargs or {}
        trajectories_output = stitch_trajectory_fragments(
            trajectories_output,
            max_gap=stitch_max_gap,
            base_distance=stitch_base_distance,
            verbose=verbose_stitching,
            **stitch_kwargs,
        )

    if len(trajectories_output) == 0:
        return [], cost_peak2peak, [], np.inf

    (
        trajectories_output,
        cost_traj2traj,
        assignment_output,
    ) = assign_trajectories(
        trajectories_output,
        trajectories_GT,
        algorithm=algo_traj2traj,
        cost_function=cost_func_traj2traj,
        verbose=verbose_assignment,
    )

    return (
        trajectories_output,
        cost_peak2peak,
        assignment_output,
        cost_traj2traj,
    )

# =============================================================================
# Trajectory utilities
# =============================================================================

# calculate center of gravity of each trajectory
def cog(trajectory):
    positions = trajectory.get_positions()
    if len(positions) == 0:
        return None
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]
    cog_row = np.mean(rows) # simple mean in rows
    cog_col = np.mean(cols) # simple mean in cols
    return (float(cog_row), float(cog_col))

def trajectories_overlap_in_time(traj1, traj2):
    return not (traj1.end_frame < traj2.start_frame or traj2.end_frame < traj1.start_frame)

def hide_immobile_trajectories(frames, trajectories, bg_mean, bg_std, threshold=0.2):
    frames_hidden = frames.copy()
    nframes = frames_hidden.shape[0]

    for traj in trajectories:

        # immobile if both principal diffusion coefficients are tiny
        D1 = traj.D1
        D2 = traj.D2

        if (not np.isnan(D1) and not np.isnan(D2) 
            and D1 < threshold and D2 < threshold):

            print(f"Hiding trajectory {traj.id} with D1={D1:.3f}, D2={D2:.3f}")

            for f in range(nframes):

                pos = traj.get_position_at_frame(f)
                if pos is None:
                    continue

                y, x = int(pos[0]), int(pos[1])

                sigma = traj.get_sigma_at_frame(f)
                if sigma is None:
                    continue

                # radius of the circle
                radius = int(math.ceil(3 * sigma))

                # bounding box
                y0 = max(0, y - radius)
                y1 = min(frames_hidden.shape[1], y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(frames_hidden.shape[2], x + radius + 1)

                # coordinate grid
                yy, xx = np.ogrid[y0:y1, x0:x1]

                # circular mask
                mask = (yy - y)**2 + (xx - x)**2 <= radius**2

                # generate background noise only for masked pixels
                patch = np.random.normal(
                    loc=bg_mean,
                    scale=bg_std,
                    size=(y1 - y0, x1 - x0)
                )
                patch = np.clip(patch, 0, 1000)

                # apply only inside circle
                frames_hidden[f, y0:y1, x0:x1][mask] = patch[mask]

    return frames_hidden

def hide_trajectories(frames, trajectories, bg_mean, bg_std):
    frames_hidden = frames.copy()
    nframes = frames_hidden.shape[0]

    for traj in trajectories:
        for f in range(nframes):

            pos = traj.get_position_at_frame(f)
            if pos is None:
                continue

            y, x = int(pos[0]), int(pos[1])

            sigma = traj.get_sigma_at_frame(f)
            if sigma is None:
                continue

            # radius of the circle
            radius = int(math.ceil(3 * sigma))

            # bounding box
            y0 = max(0, y - radius)
            y1 = min(frames_hidden.shape[1], y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(frames_hidden.shape[2], x + radius + 1)

            # coordinate grid
            yy, xx = np.ogrid[y0:y1, x0:x1]

            # circular mask
            mask = (yy - y)**2 + (xx - x)**2 <= radius**2

            # generate background noise only for masked pixels
            patch = np.random.normal(
                loc=bg_mean,
                scale=bg_std,
                size=(y1 - y0, x1 - x0)
            )
            patch = np.clip(patch, 0, 1000)

            # apply only inside circle
            frames_hidden[f, y0:y1, x0:x1][mask] = patch[mask]

    return frames_hidden


# =============================================================================
# Evaluation helpers
# =============================================================================

def compute_d_l_trajectories(frames, trajectories_GT, D_GT):
    # detection
    detected_peaks = detect_peaks(frames, threshold_abs=500, min_distance=1)
    trajectories_detection = NN_tracking_enhanced(detected_peaks, max_distance=5) # allows to pick up trajectories that do not start from the first frame
    localized_peaks = localize_peaks_with_gaussian_fitting(frames, detected_peaks, verbose=True, visualization=True)
    trajectories_localization = NN_tracking_enhanced(localized_peaks, max_distance=5)

    return trajectories_detection, trajectories_localization

# =============================================================================
# Legacy nearest-neighbor tracking methods
# =============================================================================

def NN_tracking(peaks, max_distance=5):
    """
    peaks: list of arrays, one per frame
           each peaks[f] has shape (n_peaks_f, 2)

    Returns
    -------
    trajectories: list of Trajectory objects
    """
    if len(peaks) == 0 or len(peaks[0]) == 0:
        return []

    trajectories = []
    active = []

    # initialize from frame 0
    for i, peak in enumerate(peaks[0]):
        traj = Trajectory(i, start_frame=0)
        traj.add_position(peak, frame=0)
        trajectories.append(traj)
        active.append(i)

    for f in range(1, len(peaks)):
        current_peaks = np.asarray(peaks[f])

        if len(current_peaks) == 0:
            active = []
            break

        used = np.zeros(len(current_peaks), dtype=bool)
        new_active = []

        for traj_idx in active:
            traj = trajectories[traj_idx]
            last_pos = np.array(traj.last_position())

            distances = np.linalg.norm(current_peaks - last_pos, axis=1)
            distances[used] = np.inf

            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist <= max_distance:
                traj.add_position(current_peaks[min_idx], frame=f)
                used[min_idx] = True
                new_active.append(traj_idx)

        active = new_active

        if len(active) == 0:
            break

    return trajectories

def NN_tracking_enhanced(peaks, max_distance=5, min_length=2):
    """
    Enhanced NN tracking:
    - links peaks across consecutive frames
    - allows new trajectories to start at any frame
    - trajectories end automatically when no match is found
    """
    if len(peaks) == 0:
        return []

    trajectories = []
    active = []

    for f in range(len(peaks)):
        current_peaks = np.asarray(peaks[f])

        if len(current_peaks) == 0:
            active = []
            continue

        used = np.zeros(len(current_peaks), dtype=bool)
        new_active = []

        # link existing trajectories
        for traj_idx in active:
            traj = trajectories[traj_idx]
            last_pos = np.array(traj.last_position())

            distances = np.linalg.norm(current_peaks - last_pos, axis=1)
            distances[used] = np.inf

            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist <= max_distance:
                traj.add_position(current_peaks[min_idx], frame=f)
                used[min_idx] = True
                new_active.append(traj_idx)

        # initialize new trajectories from unassigned peaks
        for i, peak in enumerate(current_peaks):
            if not used[i]:
                traj_id = len(trajectories) # new trajectory id is next available index
                traj = Trajectory(traj_id, start_frame=f) # create new trajectory with start frame
                traj.add_position(peak, frame=f) # add peak as first position with frame info
                trajectories.append(traj)
                new_active.append(traj_id)

        active = new_active

    trajectories = [traj for traj in trajectories if traj.length() >= min_length] # keep only trajectories with at least min_length positions

    return trajectories

def NN_tracking_blinking(peaks, max_distance=5, min_length=2, max_missing_frames=2):
    """
    Enhanced NN tracking:
    - links peaks across consecutive frames, with a tolerance for 1-2 frames of blinking (missing detections)
    - allows new trajectories to start at any frame
    - trajectories end automatically when no match is found above a certain tolerance of missing frames
    """
    if len(peaks) == 0:
        return []

    trajectories = []
    active = []

    for f in range(len(peaks)):
        current_peaks = np.asarray(peaks[f])

        if len(current_peaks) == 0:
            active = []
            continue

        used = np.zeros(len(current_peaks), dtype=bool)
        new_active = []

        # link existing trajectories
        for traj_idx in active:
            traj = trajectories[traj_idx]
            last_pos = np.array(traj.last_position())

            distances = np.linalg.norm(current_peaks - last_pos, axis=1)
            distances[used] = np.inf

            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist <= max_distance:
                traj.add_position(current_peaks[min_idx], frame=f)
                used[min_idx] = True
                new_active.append(traj_idx)

        # initialize new trajectories from unassigned peaks
        for i, peak in enumerate(current_peaks):
            if not used[i]:
                traj_id = len(trajectories) # new trajectory id is next available index
                traj = Trajectory(traj_id, start_frame=f) # create new trajectory with start frame
                traj.add_position(peak, frame=f) # add peak as first position with frame info
                trajectories.append(traj)
                new_active.append(traj_id)

        active = new_active

    trajectories = [traj for traj in trajectories if traj.length() >= min_length] # keep only trajectories with at least min_length positions

    return trajectories


