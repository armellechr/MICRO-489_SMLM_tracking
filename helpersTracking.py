import numpy as np
from helpersGeneration import Trajectory
from helpersAssignment import *
from helpersAssignment import assign_trajectories, cog, cost_cog
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit

# ----- LOCAL NN TRACKING -----

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


def track_peaks_to_trajectories(
    peaks,
    max_distance=5,
    min_length=5,
    algorithm='hungarian',
    cost_function=None
):
    """
    General frame-to-frame tracking using detections of the form:
        ((x, y), intensity)
    or
        (x, y)
    """
    if len(peaks) == 0:
        return [], 0.0

    trajectories = []
    active = []
    min_cost = 0.0

    for f, frame_peaks in enumerate(peaks):
        current_detections = []

        # normalize detection format
        for det in frame_peaks:
            pos = tuple(det[0])
            amp = float(det[1])
            det_sigma = float(det[2])

            current_detections.append((pos, amp, det_sigma))

        if len(current_detections) == 0:
            active = []
            continue

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

        if cost_function is None:
            cost_function = PeakToPeak.default()
        else:
            custom_cost = cost_function

        cost_matrix = compute_cost_matrix_tracks_to_detections(
            active_trajectories,
            current_detections,
            cost_function=cost_function,
            max_distance=max_distance,
        )

        gated_cost = cost_matrix.copy()

        if algorithm == 'hungarian':
            min_cost, assignment = hungarian(gated_cost)
        elif algorithm == 'local_nn':
            min_cost, assignment = local_nn_assignment(gated_cost, max_distance=max_distance)
        elif algorithm == 'global_nn':
            min_cost, assignment = global_nn_assignment(gated_cost, max_distance=max_distance)
        else:
            raise ValueError("Invalid algorithm. Choose 'hungarian', 'local_nn', or 'global_nn'.")

        used = np.zeros(len(current_detections), dtype=bool)
        new_active = []

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

        for j, det in enumerate(current_detections):
            if not used[j]:
                pos, amp, sigma = det
                traj_id = len(trajectories)
                traj = Trajectory(traj_id, start_frame=f)
                traj.add_position(tuple(pos), frame=f, intensity=amp, sigma=sigma)
                trajectories.append(traj)
                new_active.append(traj_id)

        active = new_active

    trajectories = [traj for traj in trajectories if traj.length() >= min_length]
    return trajectories, min_cost

# ----- DETECTION -----

# implement peak finder on each frame to extract particles trajectories
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

# ----- LOCALIZATION -----
def amp(sigma, A):
    return A / (2 * np.pi * sigma**2)

def gaussian_2d(xy, x0, y0, A, sigma, B):
            x, y = xy
            return B + amp(sigma, A) * np.exp(-((x - x0)**2  + (y - y0)**2) / (2 * sigma**2))

def fit_gaussian_to_peak(frame, peak, verbose=False):
    row, col = peak
    patch = frame[row-2:row+3, col-2:col+3] # 5x5 patch
    if patch.shape != (5, 5):
        return None

    patch = np.asarray(patch, dtype=float)

    x_data = np.arange(patch.shape[1])
    y_data = np.arange(patch.shape[0])
    x_data, y_data = np.meshgrid(x_data, y_data)

    # initial guesses
    sigma0 = 0.5
    peak_height0 = np.max(patch)
    A0 = 1000 * sigma0**2 * 2 * np.pi # convert to A using the formula for amplitude of Gaussian
    B0 = 100

    initial_guess = (2.0, 2.0, A0, sigma0, B0) # (x0, y0, A, sigma, B) with float center of patch as initial guess for x0 and y0

    bounds = (
        [0.0, 0.0, 0.0, 0.3, 0.0],
        [4.0, 4.0, np.inf, 4.0, np.inf]
    )

    try:
        popt, _ = curve_fit( # sortir qualité du fit
            gaussian_2d,
            (x_data.ravel(), y_data.ravel()),
            patch.ravel(),
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    x0_fit, y0_fit, A_fit, sigma_fit, B_fit = popt

    # convert back to image coordinates
    row_fit_img = (row - 2) + y0_fit
    col_fit_img = (col - 2) + x0_fit

    if verbose:
        print(
            f"[Frame {frame}] Fitted center for peak at ({row}, {col}): "
            f"({row_fit_img:.2f}, {col_fit_img:.2f}), amplitude {A_fit/(2 * np.pi * sigma_fit**2):.2f}, sigma {sigma_fit:.2f}"
        )

    # compute goodness-of-fit metrics
    yfit = gaussian_2d((x_data, y_data), *popt).reshape(patch.shape)
    residuals = patch - yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((patch - np.mean(patch))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return row_fit_img, col_fit_img, A_fit, sigma_fit, B_fit, r_squared

def localize_peaks_with_gaussian_fitting(frames, detected_peaks, r_squared_threshold=0.5, verbose=False, visualization=False, visualization_peak_idx=0):
    """ Fits a 5x5 square around detected peaks, use center pixel as initial guess, extract the center coordinates (floats) by gaussian fitting
    Args:
        frames: list of 2D numpy arrays representing the image frames
        detected_peaks: list of lists of detected peaks (as returned by detect_peaks), where each inner list corresponds to a frame and contains tuples of (x, y) coordinates of detected peaks in that frame.
        r_squared_threshold: minimum R-squared value for a fit to be considered valid.
    Returns:
        localized_peaks: list of lists of localized peaks (as returned by detect_peaks), where each inner list corresponds to a frame and contains tuples of (x, y) coordinates of localized peaks in that frame.
    """
    localized_peaks = []
    for frame_idx, (frame, peaks) in enumerate(zip(frames, detected_peaks)):
        localized_frame_peaks = []
        for peak_idx, peak in enumerate(peaks):

            fitted = fit_gaussian_to_peak(frame, peak, verbose=verbose)
            pos_fit = (fitted[0], fitted[1]) if fitted is not None else None
            A_fit = fitted[2] if fitted is not None else None
            sigma_fit = fitted[3] if fitted is not None else None
            amp_fit = A_fit / (2 * np.pi * sigma_fit**2) if A_fit is not None and sigma_fit is not None else None
            r_squared = fitted[5] if fitted is not None else None
            localized_peak = (pos_fit, amp_fit, sigma_fit, r_squared) if pos_fit is not None and amp_fit is not None and sigma_fit is not None else None

            if localized_peak is not None and r_squared is not None and r_squared >= r_squared_threshold:
                localized_frame_peaks.append(localized_peak)
            elif r_squared is None:
                print('No computed R-squared for peak at ({}, {}) in frame {}.'.format(peak[0], peak[1], frame_idx))
            elif r_squared < r_squared_threshold:
                print('Poor fit for peak at ({}, {}) in frame {}: R-squared = {:.3f} (below threshold of {}).'.format(peak[0], peak[1], frame_idx, r_squared, r_squared_threshold))

            # visualize only the first peak of the first frame
            if visualization and frame_idx == 0 and peak_idx == visualization_peak_idx and fitted is not None:
                visualize_gaussian_fit(frame, peak, fitted)

        localized_peaks.append(localized_frame_peaks)



    return localized_peaks


def visualize_gaussian_fit(frame, peak, fitted_params):
    """ For visualization purposes, shows the 5x5 patch around the detected peak, the detected peak position (red cross) 
    put at the center of its pixel, the fitted center (blue circle) and a circle representing the fitted sigma."""
    row, col = peak
    row_fit, col_fit, A_fit, sigma_fit, B_fit, r_squared = fitted_params

    half = 2

    # Center zoom around detected peak, not rounded fitted center
    r0 = int(row)
    c0 = int(col)

    r_min = max(r0 - half, 0)
    r_max = min(r0 + half + 1, frame.shape[0])
    c_min = max(c0 - half, 0)
    c_max = min(c0 + half + 1, frame.shape[1])

    zoom = frame[r_min:r_max, c_min:c_max]

    plt.figure(figsize=(6, 6))

    # Show pixels so integer+0.5 corresponds to pixel centers
    plt.imshow(
        zoom,
        cmap='gray',
        origin='upper',
        extent=[0, zoom.shape[1], zoom.shape[0], 0]
    )

    # Detected peak at center of its pixel
    peak_x = (col - c_min) + 0.5
    peak_y = (row - r_min) + 0.5

    plt.scatter(peak_x, peak_y, c='r', marker='x', label='Detected peak')

    # Fitted center in the same local coordinate system
    fit_x = (col_fit - c_min) + 0.5
    fit_y = (row_fit - r_min) + 0.5

    plt.scatter(fit_x, fit_y, c='b', marker='o', label='Fitted center')

    radius = sigma_fit
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = fit_x + radius * np.cos(theta)
    circle_y = fit_y + radius * np.sin(theta)

    plt.plot(circle_x, circle_y, 'b--', linewidth=1.5, label='σ radius')

    plt.xlim(0, zoom.shape[1])
    plt.ylim(zoom.shape[0], 0)

    plt.title(f"Gaussian fit ({half*2+1}×{half*2+1} Zoom), R²={r_squared:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()



# ----- EVALUATION OF DETECTION AND LOCALIZATION -----
def compute_d_l_trajectories(frames, trajectories_GT, D_GT):
    # detection
    detected_peaks = detect_peaks(frames, threshold_abs=500, min_distance=1)
    trajectories_detection = NN_tracking_enhanced(detected_peaks, max_distance=5) # allows to pick up trajectories that do not start from the first frame
    localized_peaks = localize_peaks_with_gaussian_fitting(frames, detected_peaks, verbose=True, visualization=True)
    trajectories_localization = NN_tracking_enhanced(localized_peaks, max_distance=5)

    return trajectories_detection, trajectories_localization
