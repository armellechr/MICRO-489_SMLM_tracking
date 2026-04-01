import numpy as np
import matplotlib.pyplot as plt
import math
from helpersGeneration import generate_frames_setD, Trajectory
from helpersTracking import NN_tracking, detect_peaks, localize_peaks_with_gaussian_fitting
from helpersAssignment import assign_trajectories
from scipy.optimize import curve_fit

# ----- MSD EXTRACTION -----

def msd_trajectory(trajectory):
    """Calculate the mean squared displacement (MSD) for a single trajectory.
    Args:
        trajectory: A Trajectory object containing positions of a single particle over time.
    Returns:
        msd: A numpy array of shape (num_steps,) containing the MSD for each time lag.
    """
    len = trajectory.length()
    positions = np.array(trajectory.get_positions())    
    msd = np.zeros(len)
    for tau in range(1,len):
        # Calculate the square of displacements for each tau time t
        displacements = np.sum((positions[tau:] - positions[:len-tau])**2, axis=1)
        msd[tau] = np.mean(displacements)  # Average displacement for the given lag
    return msd

def calculate_msd_trajectories(trajectories):
    """Calculate the mean squared displacement (MSD) for a list of trajectories.
    Args:
        trajectories: A list of Trajectory objects, each containing positions of a single particle over time.
    Returns:
        msd: A dictionary where keys are trajectory IDs and values are numpy arrays of shape (num_steps,) containing the MSD for each time lag.
    """
    for traj in trajectories:
        if traj.id == None:
            traj.msd = []
        else:
            msd_traj = msd_trajectory(traj)
            traj.msd = msd_traj
            

# # plot the MSD according to the assigned GT id
# def plot_msd_vs_trajectory_msd(trajectories_new, trajectories_GT):
#     msd_new = [traj.msd for traj in trajectories_new]
#     msd_trajectory = [traj.msd for traj in trajectories_GT]
#     plt.figure(figsize=(15, 5))
#     for i in range(len(msd_new)):
#         plt.subplot(2, len(msd_new)//2 + len(msd_new) % 2, i+1)
#         if i not in range(len(msd_trajectory)):
#             plt.plot(msd_new[i], label='Experimental MSD')
#             plt.xlabel('Time lag (tau)')
#             plt.ylabel('MSD')
#             plt.title(f'ID: {trajectories_new[i].id} (No GT match)')
#         else:
#             plt.plot(msd_new[i], label='Experimental MSD')
#             plt.plot(msd_trajectory[i], label='Trajectory MSD')
#             plt.xlabel('Time lag (tau)')
#             plt.ylabel('MSD')
#             plt.title(f'GT ID: {trajectories_GT[i].id}')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_msd_vs_trajectory_msd(trajectories_new, trajectories_GT):
    """
    Plot each detected trajectory MSD against the GT trajectory MSD
    matched by trajectory id.
    Plot only trajectories that were assigned a GT id (i.e. traj.id is not None) and for which the GT id exists in the GT trajectories list.

    Assumes:
        - each traj in trajectories_new has traj.id set to the matched GT id
        - each traj in trajectories_GT has its own GT id in traj.id
        - traj.msd contains the MSD array/list
    """
    # map GT id -> GT trajectory
    gt_by_id = {traj.id: traj for traj in trajectories_GT if traj.id is not None}

    n = len(trajectories_new)
    if n == 0:
        print("No trajectories to plot.")
        return

    ncols = 2
    nrows = math.ceil(n / ncols)

    plt.figure(figsize=(12, 4 * nrows))

    # mask to keep only trajectories with an ID different from -1
    trajectories_new_assigned = [traj for traj in trajectories_new if traj.id is not None and traj.id != -1]

    # plot only trajectories with a GT match
    for i, traj_new in enumerate(trajectories_new_assigned):
        plt.subplot(nrows, ncols, i + 1)

        msd_new = traj_new.msd
        gt_id = traj_new.id

        # Plot detected / experimental MSD
        if msd_new is not None and len(msd_new) > 0:
            plt.plot(msd_new, label='Experimental MSD')

        # Plot matched GT MSD if id exists and is found
        if gt_id is not None and gt_id in gt_by_id:
            msd_gt = gt_by_id[gt_id].msd
            if msd_gt is not None and len(msd_gt) > 0:
                plt.plot(msd_gt, ':', label='Ground-truth MSD')
            plt.title(f'New traj matched to GT ID {gt_id}')
        else:
            plt.title(f'New traj ID {gt_id} (No GT match)')

        plt.xlabel('Time lag (tau)')
        plt.ylabel('MSD')
        plt.legend()

    plt.tight_layout()
    plt.show()

# ----- DIFFUSION COEFFICIENT ESTIMATION -----

def estimateDfromMSD(msd, time_range):
    """
    Estimates the diffusion coefficient (D) from the mean squared displacement (MSD) 
    using linear regression without an intercept.
    
    Parameters:
    - msd (numpy array): 1D array of MSD values.
    - time_range (numpy array): 1D array of time values corresponding to MSD.
    
    Returns:
    - D_estimated (float): Estimated diffusion coefficient.
    """
    # Solve the least squares problem to fit a line through the origin (y = slope * x)
    slope, = np.linalg.lstsq(time_range.reshape(-1, 1), msd, rcond=None)[0]
    
    # Diffusion coefficient is the slope divided by 4
    D_estimated = slope / 4 # FACTEUR 2 EN TROP
    return D_estimated

def estimateDfromTrajectories(trajectories):
    D_list = []
    for traj in trajectories:
        D_traj = estimateDfromMSD(traj.msd, np.arange(len(traj.msd)))
        traj.D_trajectory = D_traj
        D_list.append(D_traj)
    return D_list

# def estimateDfromTrajectories_safer(trajectories, type='trajectory'):
#     D_list = []
    
#     for i, traj in enumerate(trajectories):
#         try:
#             msd = np.asarray(traj.msd, dtype=float)

#             if len(msd) < 2:
#                 print(f"Trajectory {i}: msd too short")
#                 D_list.append(np.nan)
#                 if type == 'trajectory':
#                     traj.D_trajectory = np.nan
#                 elif type == 'detection':
#                     traj.D_detection = np.nan
#                 elif type == 'localization':
#                     traj.D_localization = np.nan
#                 continue

#             if not np.all(np.isfinite(msd)):
#                 print(f"Trajectory {i}: msd contains nans or infs")
#                 D_list.append(np.nan)
#                 if type == 'trajectory':
#                     traj.D_trajectory = np.nan
#                 elif type == 'detection':
#                     traj.D_detection = np.nan
#                 elif type == 'localization':
#                     traj.D_localization = np.nan
#                 continue

#             tau = np.arange(len(msd), dtype=float)
#             D_traj = estimateDfromMSD(msd, tau)
#             if type == 'trajectory':
#                 traj.D_trajectory = D_traj
#             elif type == 'detection':
#                 traj.D_detection = D_traj
#             elif type == 'localization':
#                 traj.D_localization = D_traj
#             D_list.append(D_traj)

#         except Exception as e:
#             print(f"Trajectory {i}: error -> {e}")
#             if type == 'trajectory':
#                 traj.D_trajectory = np.nan
#             elif type == 'detection':
#                 traj.D_detection = np.nan
#             elif type == 'localization':
#                 traj.D_localization = np.nan
#             D_list.append(np.nan)

#     return D_list

def estimateDfromTrajectories_safer(trajectories, mode='trajectory'):
    D_list = []

    for i, traj in enumerate(trajectories):
        try:
            msd = np.asarray(traj.msd, dtype=float)

            if msd.ndim != 1 or len(msd) < 2 or not np.all(np.isfinite(msd)):
                D_val = np.nan
            else:
                tau = np.arange(len(msd), dtype=float)
                D_val = estimateDfromMSD(msd, tau)

            # store in correct attribute
            if mode == 'trajectory':
                traj.D_trajectory = D_val
            elif mode == 'detection':
                traj.D_detection = D_val
            elif mode == 'localization':
                traj.D_localization = D_val

            D_list.append(D_val)

        except Exception as e:
            print(f"Trajectory {i}: error -> {e}")
            D_val = np.nan

            if mode == 'trajectory':
                traj.D_trajectory = D_val
            elif mode == 'detection':
                traj.D_detection = D_val
            elif mode == 'localization':
                traj.D_localization = D_val

            D_list.append(D_val)

    return D_list

# ----- MAE CALCULATION -----

def compute_MAE(truth, estimate):
    if truth is None or estimate is None:
        return None
    return abs(truth - estimate)

def compute_D_trajectory(trajectories_GT, D_GT):

    # compute and set MSD in trajectory objects
    calculate_msd_trajectories(trajectories_GT)
    
    # compute and print D_trajectory for each trajectory
    D_trajectory = estimateDfromTrajectories_safer(trajectories_GT, 'trajectory')
    print(f"D_trajectory: \n[{', '.join(f'{float(d):.3f}' for d in D_trajectory)}]")
    
    return D_trajectory

def compute_MAE_trajectory(trajectories_GT, D_GT):

    # compute and set MSD in trajectory objects
    calculate_msd_trajectories(trajectories_GT)
    
    # compute and print D_trajectory for each trajectory
    D_trajectory = estimateDfromTrajectories_safer(trajectories_GT, 'trajectory')
    print(f"D_trajectory: \n[{', '.join(f'{float(d):.3f}' for d in D_trajectory)}]")
    
    # compute and print MAE for D_trajectory
    MAE_trajectory = np.abs(np.array(D_trajectory) - np.array(D_GT))
    print(f"MAE for D_trajectory: \n[{', '.join(f'{float(d):.3f}' for d in MAE_trajectory)}]")  

    return MAE_trajectory

# def compute_MAE_detection(trajectories_detection, D_GT):

#     # compute and set MSD in trajectory objects
#     calculate_msd_trajectories(trajectories_detection)

#     # compute and print D_detection for each trajectory
#     D_detection = estimateDfromTrajectories_safer(trajectories_detection, 'detection')
#     print(f"D_detection: \n[{', '.join(f'{float(d):.3f}' for d in D_detection)}]")
    
#     # compute and print MAE for D_detection based on D_GT indices
#     MAE_detection = {}
#     for traj in trajectories_detection:
#         if traj.id is not None and traj.id < len(D_GT):
#             mae = compute_MAE(traj.D_detection, D_GT[traj.id])
#             MAE_detection[traj.id] = mae
#         else:
#             print(f"No valid GT match for detected trajectory {traj.id}.")
#             MAE_detection[traj.id] = None
    
def compute_MAE_detection(trajectories_detection, D_GT):
    calculate_msd_trajectories(trajectories_detection)

    estimateDfromTrajectories_safer(trajectories_detection, 'detection')

    # initialize list aligned with GT
    D_detection = [np.nan] * len(D_GT)

    for traj in trajectories_detection:
        if traj.id is not None and 0 <= traj.id <= len(D_GT) - 1:
            D_detection[traj.id] = traj.D_detection

    print("D_detection:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D_detection])

    MAE_detection = np.abs(np.array(D_detection) - np.array(D_GT))

    print("MAE for D_detection:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in MAE_detection])
    
    return MAE_detection

def compute_D_detection(trajectories_detection, D_GT):
    calculate_msd_trajectories(trajectories_detection)

    estimateDfromTrajectories_safer(trajectories_detection, 'detection')

    # initialize list aligned with GT
    D_detection = [np.nan] * len(D_GT)

    for traj in trajectories_detection:
        if traj.id is not None and 0 <= traj.id <= len(D_GT) - 1:
            D_detection[traj.id] = traj.D_detection

    print("D_detection:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D_detection])

    return D_detection

# def compute_MAE_localization(trajectories_localization, D_GT):

#     # compute and set MSD in trajectory objects
#     calculate_msd_trajectories(trajectories_localization)

#     # compute and print D_localization for each trajectory
#     D_localization = estimateDfromTrajectories_safer(trajectories_localization, 'localization')
#     print(f"D_localization: \n[{', '.join(f'{float(d):.3f}' for d in D_localization)}]")

#     # compute and print MAE for D_localization based on D_GT indices
#     MAE_localization = {}
#     for traj in trajectories_localization:
#         if traj.id is not None and traj.id < len(D_GT):
#             mae = compute_MAE(traj.D_localization, D_GT[traj.id])
#             MAE_localization[traj.id] = mae
#         else:
#             print(f"No valid GT match for detected trajectory {traj.id}.")
#             MAE_localization[traj.id] = None

#     print(f"MAE for D_localization: \n[{', '.join(f'ID {id}: {float(mae):.3f}' if mae is not None else f'ID {id}: None' for id, mae in MAE_localization.items())}]")
    
#     return MAE_localization

def compute_MAE_localization(trajectories_localization, D_GT):
    calculate_msd_trajectories(trajectories_localization)

    estimateDfromTrajectories_safer(trajectories_localization, 'localization')

    D_localization = [np.nan] * len(D_GT)

    for traj in trajectories_localization:
        if traj.id is not None and 0 <= traj.id <= len(D_GT) - 1:
            D_localization[traj.id] = traj.D_localization

    print("D_localization:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D_localization])

    MAE_localization = np.abs(np.array(D_localization) - np.array(D_GT))

    print("MAE for D_localization:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in MAE_localization])

    return MAE_localization

def compute_D_localization(trajectories_localization, D_GT):
    calculate_msd_trajectories(trajectories_localization)

    estimateDfromTrajectories_safer(trajectories_localization, 'localization')

    D_localization = [np.nan] * len(D_GT)

    for traj in trajectories_localization:
        if traj.id is not None and 0 <= traj.id <= len(D_GT) - 1:
            D_localization[traj.id] = traj.D_localization

    print("D_localization:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D_localization])

    return D_localization

def compute_MAEs_setD(F, N, D):

    frames, trajectories_GT, D_GT = generate_frames_setD(F, N, D)

    # traj_GT
    MAE_trajectory = compute_MAE_trajectory(trajectories_GT, D_GT)
    # detection
    detected_peaks = detect_peaks(frames)
    trajectories_detection = NN_tracking(detected_peaks, max_distance=5)
    trajectories_detection, min_cost, _ = assign_trajectories(trajectories_detection, trajectories_GT, algorithm='hungarian')
    MAE_detection = compute_MAE_detection(trajectories_detection, D_GT)
    # localization
    localized_peaks = localize_peaks_with_gaussian_fitting(frames, detected_peaks)
    trajectories_localization = NN_tracking(localized_peaks, max_distance=5)
    trajectories_localization, min_cost, _ = assign_trajectories(trajectories_localization, trajectories_GT, algorithm='hungarian')
    MAE_localization = compute_MAE_localization(trajectories_localization, D_GT)
    
    return MAE_trajectory, MAE_detection, MAE_localization

def scatter_mae_cloud(D, MAE_array, label, color='blue'):
    D = np.asarray(D)
    MAE_array = np.asarray(MAE_array, dtype=float)
    # Repeat each D for all trajectory-wise MAEs in that row
    x = np.repeat(D, MAE_array.shape[1])
    y = MAE_array.ravel()

    # Remove NaNs (and any infs)
    mask = np.isfinite(y)
    plt.scatter(x[mask], y[mask], label=label, color=color)

def lin_model(a, x):
    return a * x

def fit_line_mae(D, MAE_array):
    D = np.asarray(D)
    MAE_array = np.asarray(MAE_array, dtype=float)

    x = np.repeat(D, MAE_array.shape[1])
    y = MAE_array.ravel()

    mask = np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 2:
        print("Not enough valid points to fit a line.")
        return None
    
    # fit using curve_fit on lin_model
    try:
        popt, _ = curve_fit(lin_model, x_fit, y_fit)
        return popt  # returns [a]
    except Exception as e:
        print(f"Error fitting line: {e}")
        return None
    
def fit_line_D(D, D_array):
    D = np.asarray(D, dtype=float)
    D_array = np.asarray(D_array, dtype=float)

    x = D.ravel()
    y = D_array.ravel()

    mask = np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < 2:
        print("Not enough valid points to fit a line.")
        return None

    try:
        popt, _ = curve_fit(lin_model, x_fit, y_fit)
        return popt
    except Exception as e:
        print(f"Error fitting line: {e}")
        return None



def plot_MAEs_vs_D(D, MAE_t_array, MAE_d_array, MAE_l_array):

    plt.figure(figsize=(10, 6))

    scatter_mae_cloud(D, MAE_t_array, 'Trajectory MAE', color='blue')
    scatter_mae_cloud(D, MAE_d_array, 'Detection MAE', color='green')
    scatter_mae_cloud(D, MAE_l_array, 'Localization MAE', color='red')

    # linear fit
    coeff_t = fit_line_mae(D, MAE_t_array)
    coeff_d = fit_line_mae(D, MAE_d_array)
    coeff_l = fit_line_mae(D, MAE_l_array) # shape of coeff is (1,) since we only have one parameter a

    if coeff_t is not None:
        a_t = coeff_t[0]
        x_fit = np.array(D)
        y_fit_t = lin_model(a_t, x_fit)
        plt.plot(x_fit, y_fit_t, color='blue', linestyle='--', label=f'Trajectory fit: y={a_t:.3f}x')
    if coeff_d is not None:
        a_d = coeff_d[0]
        x_fit = np.array(D)
        y_fit_d = lin_model(a_d, x_fit)
        plt.plot(x_fit, y_fit_d, color='green', linestyle='--', label=f'Detection fit: y={a_d:.3f}x')
    if coeff_l is not None:
        a_l = coeff_l[0]
        x_fit = np.array(D)
        y_fit_l = lin_model(a_l, x_fit)
        plt.plot(x_fit, y_fit_l, color='red', linestyle='--', label=f'Localization fit: y={a_l:.3f}x')

    plt.xlabel('D')
    plt.ylabel('MAE')
    plt.title('MAE vs. D')
    plt.legend()
    plt.grid()
    plt.show()

def plot_mae_comparison(MAE_local, MAE_global, MAE_hungarian, D_type="Detection"):
    # mask nans
    mae_local = np.ma.masked_invalid(MAE_local)
    mae_global = np.ma.masked_invalid(MAE_global)
    mae_hungarian = np.ma.masked_invalid(MAE_hungarian)

    n = len(mae_local)
    x = np.arange(n) 

    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, mae_local, width, label='Local NN')
    plt.bar(x,         mae_global, width, label='Global NN')
    plt.bar(x + width, mae_hungarian, width, label='Hungarian')

    plt.xlabel("Trajectory ID")
    plt.ylabel("MAE")
    plt.title(f"{D_type} - MAE Comparison across assignment algorithms")
    plt.xticks(x)  # show all trajectory IDs
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def compute_Ds_setD(F, N, D):

    frames, trajectories_GT, D_GT = generate_frames_setD(F, N, D)

    # traj_GT
    D_trajectory = compute_D_trajectory(trajectories_GT, D_GT)
    # detection
    detected_peaks = detect_peaks(frames)
    trajectories_detection = NN_tracking(detected_peaks, max_distance=5)
    trajectories_detection, min_cost, _ = assign_trajectories(trajectories_detection, trajectories_GT, algorithm='hungarian')
    D_detection = compute_D_detection(trajectories_detection, D_GT)
    # localization
    localized_peaks = localize_peaks_with_gaussian_fitting(frames, detected_peaks)
    trajectories_localization = NN_tracking(localized_peaks, max_distance=5)
    trajectories_localization, min_cost, _ = assign_trajectories(trajectories_localization, trajectories_GT, algorithm='hungarian')
    D_localization = compute_D_localization(trajectories_localization, D_GT)

    return D_trajectory, D_detection, D_localization

def plot_Ds_vs_D_GT(D_t, D_d, D_l, D):
    # Build D_GT as a 2D array matching the shape of D_t
    D_GT = np.tile(np.array(D).reshape(-1, 1), (1, len(D_t[0])))

    plt.figure(figsize=(10, 6))
    plt.scatter(D_GT, D_t, label='D_trajectory', color='blue')
    plt.scatter(D_GT, D_d, label='D_detection', color='green')
    plt.scatter(D_GT, D_l, label='D_localization', color='red')

    # Fit lines
    coeff_t = fit_line_D(D_GT, D_t)
    coeff_d = fit_line_D(D_GT, D_d)
    coeff_l = fit_line_D(D_GT, D_l)

    if coeff_t is not None:
        a = coeff_t[0]
        plt.plot(D, a * np.array(D), '--', color='blue')

    if coeff_d is not None:
        a = coeff_d[0]
        plt.plot(D, a * np.array(D), '--', color='green')

    if coeff_l is not None:
        a = coeff_l[0]
        plt.plot(D, a * np.array(D), '--', color='red')

    plt.plot(D, D, '--', color='black', label='D_GT')

    plt.xlabel('D_GT')
    plt.ylabel('Estimated D')
    plt.title('Estimated D vs D_GT')
    plt.legend()
    plt.grid()
    plt.show()

    