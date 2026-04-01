import numpy as np
from helpersGeneration import Trajectory
from helpersAssignment import assign_trajectories, cog, cost_cog
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit

# ----- LOCAL NN TRACKING -----

# based on the detected peaks -> tracking algorithm to link the peaks across frames and reconstruct the trajectories of the particles as a list of Trajectories objects
# chosen method = nearest neighbor algorithm (each peak linked to the closest peak in the next frame within a certain distance threshold)
# def NN_tracking(peaks, max_distance=5):
#     """
#     peaks: list of arrays, one per frame
#            each peaks[f] has shape (n_peaks_f, 2)
#     max_distance: maximum allowed jump between consecutive frames

#     Returns
#     -------
#     trajectories: list of Trajectory objects
#     """
#     if len(peaks) == 0 or len(peaks[0]) == 0: # if no peaks in first frame, cannot initialize any trajectory
#         return []

#     # initialize one trajectory per peak in first frame
#     trajectories = []
#     active = [] # indices of trajectories still being tracked

#     for i, peak in enumerate(peaks[0]):
#         traj = Trajectory(i)
#         traj.add_position(peak)
#         trajectories.append(traj)
#         active.append(i)

#     # process subsequent frames
#     for f in range(1, len(peaks)):
#         current_peaks = np.asarray(peaks[f])

#         if len(current_peaks) == 0:
#             active = []
#             break

#         used = np.zeros(len(current_peaks), dtype=bool)
#         new_active = []

#         for traj_idx in active:
#             traj = trajectories[traj_idx]
#             last_pos = np.array(traj.last_position())

#             distances = np.linalg.norm(current_peaks - last_pos, axis=1)

#             # ignore already assigned peaks
#             distances[used] = np.inf

#             min_idx = np.argmin(distances)
#             min_dist = distances[min_idx]

#             if min_dist <= max_distance:
#                 traj.add_position(current_peaks[min_idx]) # add_position method of Trajectory class
#                 used[min_idx] = True
#                 new_active.append(traj_idx)
#             # else: trajectory ends here

#         active = new_active

#         if len(active) == 0:
#             break

#     return trajectories

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

# def NN_tracking_enchanced(peaks, max_distance=5):
#     """
#     Enhanced version that allows to pick up trajectories that do not start from the first frame, 
#     by also initializing new trajectories from unassigned peaks in each frame.
#     """
#     if len(peaks) == 0: # if no peaks at all, cannot initialize any trajectory
#         return []
    
#     trajectories = [] # list of trajectory indices that are still alive
#     active = []
    
#     for f in range(len(peaks)): # for each frame
#         current_peaks = np.asarray(peaks[f]) # convert to numpy array for easier distance calculations
    
#         if len(current_peaks) == 0: # if no peaks in current frame, all active trajectories end here
#             active = []
#             continue
    
#         used = np.zeros(len(current_peaks), dtype=bool)
#         new_active = []
    
#         # first try to link existing trajectories
#         for traj_idx in active: # for all active trajectories
#             traj = trajectories[traj_idx]
#             last_pos = np.array(traj.last_position()) # get last position of trajectory as numpy array
    
#             distances = np.linalg.norm(current_peaks - last_pos, axis=1) # compute distances to all peaks in current frame
#             distances[used] = np.inf # ignore already assigned peaks by setting their distance to infinity
    
#             min_idx = np.argmin(distances) # find index of closest peak
#             min_dist = distances[min_idx] # get distance to closest peak
    
#             if min_dist <= max_distance: # distance criterion for linking
#                 traj.add_position(current_peaks[min_idx]) # add peak to trajectory
#                 used[min_idx] = True # mark this peak as used
#                 new_active.append(traj_idx) # keep trajectory active for next frame
    
#         # then initialize new trajectories from unassigned peaks
#         for i, peak in enumerate(current_peaks):
#             if not used[i]: # for all unused peaks
#                 traj_id = len(trajectories) # new trajectory id is next available index
#                 traj = Trajectory(traj_id) # create new trajectory
#                 traj.add_position(peak) # add peak as first position
#                 trajectories.append(traj) # add to list of trajectories
#                 new_active.append(traj_id) # mark new trajectory as active
    
#         active = new_active # update active trajectories for next frame
    
#     # discard trajectories that have only one position (could be noise)
#     trajectories = [traj for traj in trajectories if len(traj.get_positions()) > 1]
    
#     return trajectories

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


# ----- DETECTION -----

# implement peak finder on each frame to extract particles trajectories
def detect_peaks(frames, threshold_abs=700, min_distance=1):    
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

# nearest neighbor association between detected and GT trajectories, label detected trajectories with closest GT id
# def label_trajectories_from_GT(trajectories_new, trajectories_GT, max_distance=10):
#     cog_GT = [cog(traj) for traj in trajectories_GT]
#     used_GT_idx = []
#     for traj in trajectories_new:
#         cog_traj = cog(traj)
#         if cog_traj is None:
#             traj.id = None
#             continue
#         distances = [np.linalg.norm(np.array(cog_traj) - np.array(cog_gt)) for cog_gt in cog_GT]
#         for idx in used_GT_idx:
#             distances[idx] = np.inf # to prevent multiple assignments of the same GT trajectory
#         closest_GT_idx = np.argmin(distances)
#         if distances[closest_GT_idx] > max_distance:
#             traj.set_id(None)
#             print('No GT cog within max distance for', cog_traj, '-> id None')
#         else:
#             traj.set_id(trajectories_GT[closest_GT_idx].id)
#             used_GT_idx.append(closest_GT_idx)
#             print('Assigned', cog_traj, 'to GT', cog_GT[closest_GT_idx], '-> id', traj.id)

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
        popt, _ = curve_fit(
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
            f"Fitted center for peak at ({row}, {col}): "
            f"({row_fit_img:.2f}, {col_fit_img:.2f}), amplitude {A_fit/(2 * np.pi * sigma_fit**2):.2f}, sigma {sigma_fit:.2f}"
        )

    return row_fit_img, col_fit_img, A_fit, sigma_fit, B_fit

def localize_peaks_with_gaussian_fitting(frames, detected_peaks, verbose=False, visualization=False):
    """ Fits a 5x5 square around detected peaks, use center pixel as initial guess, extract the center coordinates (floats) by gaussian fitting
    Args:
        frames: list of 2D numpy arrays representing the image frames
        detected_peaks: list of lists of detected peaks (as returned by detect_peaks), where each inner list corresponds to a frame and contains tuples of (x, y) coordinates of detected peaks in that frame.
    Returns:
        localized_peaks: list of lists of localized peaks (as returned by detect_peaks), where each inner list corresponds to a frame and contains tuples of (x, y) coordinates of localized peaks in that frame.
    """
    localized_peaks = []
    for frame_idx, (frame, peaks) in enumerate(zip(frames, detected_peaks)):
        localized_frame_peaks = []
        for peak_idx, peak in enumerate(peaks):

            fitted = fit_gaussian_to_peak(frame, peak, verbose=verbose)
            localized_peak = (fitted[0], fitted[1]) if fitted is not None else None

            if localized_peak is not None:
                localized_frame_peaks.append(localized_peak)

            # visualize only the first peak of the first frame
            if visualization and frame_idx == 0 and peak_idx == 0 and fitted is not None:
                visualize_gaussian_fit(frame, peak, fitted)

        localized_peaks.append(localized_frame_peaks)

    return localized_peaks

# def visualize_gaussian_fit(frame, peak, fitted_params):
#     row, col = peak
#     x0_fit, y0_fit, A_fit, sigma_fit, B_fit = fitted_params

#     # --- Define zoom window (5x5 around fitted center) ---
#     half = 2  # 2 pixels on each side → 5×5 window

#     r0 = int(round(x0_fit))
#     c0 = int(round(y0_fit))

#     r_min = max(r0 - half, 0)
#     r_max = min(r0 + half + 1, frame.shape[0])
#     c_min = max(c0 - half, 0)
#     c_max = min(c0 + half + 1, frame.shape[1])

#     zoom = frame[r_min:r_max, c_min:c_max]

#     # --- Plot ---
#     plt.figure(figsize=(6, 6))
#     plt.imshow(zoom, cmap='gray', extent=[c_min, c_max, r_max, r_min])

#     # Detected peak (if inside zoom)
#     if r_min <= row < r_max and c_min <= col < c_max:
#         plt.scatter(col, row, c='r', marker='x', label='Detected Peak')

#     # Fitted center
#     plt.scatter(y0_fit, x0_fit, c='b', marker='o', label='Fitted Center')

#     # --- Draw sigma-based circle ---
#     k = 1.0  # 1-sigma radius; change to 2 or 3 if you want a larger outline
#     radius = k * sigma_fit

#     theta = np.linspace(0, 2*np.pi, 200)
#     circle_x = y0_fit + radius * np.cos(theta)
#     circle_y = x0_fit + radius * np.sin(theta)

#     plt.plot(circle_x, circle_y, 'b--', linewidth=1.5, label=f'{k}σ circle')

#     plt.title(f"Gaussian fit ({half*2+1}×{half*2+1} Zoom)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def visualize_gaussian_fit(frame, peak, fitted_params):
    row, col = peak
    row_fit, col_fit, A_fit, sigma_fit, B_fit = fitted_params

    half = 2

    r0 = int(round(row_fit))
    c0 = int(round(col_fit))

    r_min = max(r0 - half, 0)
    r_max = min(r0 + half + 1, frame.shape[0])
    c_min = max(c0 - half, 0)
    c_max = min(c0 + half + 1, frame.shape[1])

    zoom = frame[r_min:r_max, c_min:c_max]

    plt.figure(figsize=(6, 6))
    plt.imshow(zoom, cmap='gray', extent=[c_min, c_max, r_max, r_min])

    if r_min <= row < r_max and c_min <= col < c_max:
        plt.scatter(col, row, c='r', marker='x', label='Detected Peak')

    plt.scatter(col_fit, row_fit, c='b', marker='o', label='Fitted Center')

    radius = sigma_fit
    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = col_fit + radius * np.cos(theta)
    circle_y = row_fit + radius * np.sin(theta)

    plt.plot(circle_x, circle_y, 'b--', linewidth=1.5, label='1σ circle')

    plt.title(f"Gaussian fit ({half*2+1}×{half*2+1} Zoom)")
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
