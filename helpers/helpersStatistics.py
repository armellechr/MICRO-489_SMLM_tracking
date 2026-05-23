# =============================================================================
# Imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

# =============================================================================
# Scalar MSD (isotropic only))
# =============================================================================

def calculateMSDtrajectory(trajectory):
    """Calculate the mean squared displacement (MSD) for a single trajectory.
    Args:
        trajectory: A Trajectory object containing positions of a single particle over time.
    Returns:
        MSD: A numpy array of shape (num_steps,) containing the MSD for each time lag.
    """
    L = trajectory.length()
    positions = np.array(trajectory.get_positions(), dtype=float)

    MSD = np.zeros(L)

    for tau in range(1, L):
        displacements = positions[tau:] - positions[:L - tau]
        squared_displacements = np.sum(displacements**2, axis=1)
        MSD[tau] = np.mean(squared_displacements)

    return MSD

def calculateMSDtrajectories(trajectories):
    """Calculate the mean squared displacement (MSD) for a list of trajectories.
    Args:
        trajectories: A list of Trajectory objects, each containing positions of a single particle over time.
    Returns:
        MSD: A dictionary where keys are trajectory IDs and values are numpy arrays of shape (num_steps,) containing the MSD for each time lag.
    """
    MSD_list = []
    for traj in trajectories:
        if traj.id == None:
            traj.MSD = []
        else:
            MSD_traj = calculateMSDtrajectory(traj)
            traj.MSD = MSD_traj
            MSD_list.append(MSD_traj)

    return MSD_list

# =============================================================================
# Diffusion tensor estimation
# =============================================================================

def displacement_covariance(positions, tau):
    """
    Compute the 2×2 displacement covariance matrix for a given lag tau.
    """
    L = len(positions)
    if L <= tau:
        return np.full((2, 2), np.nan)

    displacements = positions[tau:] - positions[:L - tau]
    C = np.mean([np.outer(dr, dr) for dr in displacements], axis=0)
    return C

def diffusion_tensor_from_covariance(C, dt, tau):
    return C / (2 * tau * dt)


def estimateDfromTrajectory(
    trajectory,
    dt=1.0,
    tau=1,
):
    """
    Estimate D1, D2, theta from a trajectory using the displacement covariance tensor.

    Works for both isotropic and anisotropic diffusion.

    For isotropic diffusion:
        D1 ≈ D2

    For anisotropic diffusion:
        D1 > D2 and theta gives the main axis.
    """
    positions = np.asarray(trajectory.get_positions(), dtype=float)

    C = displacement_covariance(positions, tau)

    D_tensor = diffusion_tensor_from_covariance(C, dt, tau)

    eigvals, eigvecs = np.linalg.eigh(D_tensor)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    D1 = float(eigvals[0])
    D2 = float(eigvals[1])

    principal_vec = eigvecs[:, 0]
    theta = float(np.arctan2(principal_vec[1], principal_vec[0]) % np.pi)

    trajectory.D_tensor = D_tensor
    trajectory.D1 = D1
    trajectory.D2 = D2
    trajectory.theta = theta

    return D1, D2, theta, D_tensor

# list-level
def estimateDfromTrajectories(
    trajectories,
    dt=1.0,
    tau=1,
):
    """
    Estimate diffusion tensor parameters for a list of trajectories.
    """
    results = []

    for traj in trajectories:
        D1, D2, theta, D_tensor = estimateDfromTrajectory(
            traj,
            dt=dt,
            tau=tau,
        )

        results.append({
            "id": traj.id,
            "D1": D1,
            "D2": D2,
            "theta": theta,
            "anisotropy": D2 / D1 if np.isfinite(D1) and D1 > 0 else np.nan,
            "D_tensor": D_tensor,
        })

    return results

def tensor_to_scalar_D(D1, D2):
    return 0.5 * (np.asarray(D1) + np.asarray(D2))

# =============================================================================
# Scalar diffusion estimation from MSD
# =============================================================================

# def estimateDfromMSD(msd, time_range):
#     """
#     Estimates the diffusion coefficient (D) from the mean squared displacement (MSD) 
#     using linear regression without an intercept.
    
#     Parameters:
#     - msd (numpy array): 1D array of MSD values.
#     - time_range (numpy array): 1D array of time values corresponding to MSD.
    
#     Returns:
#     - D_estimated (float): Estimated diffusion coefficient.
#     """
#     # Solve the least squares problem to fit a line through the origin (y = slope * x)
#     slope, = np.linalg.lstsq(time_range.reshape(-1, 1), msd, rcond=None)[0]
    
#     # Diffusion coefficient is the slope divided by 4
#     D_estimated = slope / 4 # FACTEUR 2 EN TROP
#     return D_estimated

# def estimateDfromTrajectories(trajectories, mode='FR'):
#     D_list = []

#     for i, traj in enumerate(trajectories):
#         try:
#             msd = np.asarray(traj.msd, dtype=float)

#             if msd.ndim != 1 or len(msd) < 2 or not np.all(np.isfinite(msd)):
#                 D_val = np.nan
#             else:
#                 tau = np.arange(len(msd), dtype=float)
#                 D_val = estimateDfromMSD(msd, tau)

#             if mode == 'HR':
#                 traj.D_HR = D_val
#             elif mode == 'FR':
#                 traj.D_FR = D_val
#             elif mode == 'detection':
#                 traj.D_detection = D_val
#             elif mode == 'localization':
#                 traj.D_localization = D_val

#             D_list.append(D_val)

#         except Exception as e:
#             print(f"Trajectory {i}: error -> {e}")
#             D_val = np.nan

#             if mode == 'HR':
#                 traj.D_HR = D_val
#             elif mode == 'FR':
#                 traj.D_FR = D_val
#             elif mode == 'detection':
#                 traj.D_detection = D_val
#             elif mode == 'localization':
#                 traj.D_localization = D_val

#             D_list.append(D_val)

#     return D_list

# =============================================================================
# Ground-truth and estimated parameter extraction
# =============================================================================

def get_traj_params(trajectories, nparticles=None):
    """Extracts D1, D2, theta for a list of trajectories."""
    if nparticles is None:
        nparticles = len(trajectories)

    D1 = []
    D2 = []
    theta = []

    for traj in trajectories:

        D1.append(traj.D1)
        D2.append(traj.D2)
        theta.append(traj.theta)

    return D1, D2, theta

def get_traj_params_by_id(trajectories, id=None):
    """Extracts D1, D2, theta for a single trajectory"""
    for traj in trajectories:
        if traj.id == id:
            return traj.D1, traj.D2, traj.theta

    raise ValueError(f"No trajectory found with id {id}")

def get_traj_params_aligned(trajectories_list):
    """Extracts D1, D2, theta for a list of trajectories, aligned by trajectory id across multiple lists."""
    # Create a mapping from trajectory ID to its parameters for each list
    params_by_id = {}

    for trajectories in trajectories_list:
        for traj in trajectories:
            if traj.id is not None:
                if traj.id not in params_by_id:
                    params_by_id[traj.id] = {}
                params_by_id[traj.id][traj] = {
                    "D1": traj.D1,
                    "D2": traj.D2,
                    "theta": traj.theta,
                }

    # Now we can create aligned lists of parameters based on the IDs
    aligned_params = {traj: {"D1": [], "D2": [], "theta": []} for traj in trajectories_list[0]}

    for traj in aligned_params.keys():
        for trajectories in trajectories_list:
            if traj in params_by_id and traj in params_by_id[traj]:
                aligned_params[traj]["D1"].append(params_by_id[traj][traj]["D1"])
                aligned_params[traj]["D2"].append(params_by_id[traj][traj]["D2"])
                aligned_params[traj]["theta"].append(params_by_id[traj][traj]["theta"])
            else:
                aligned_params[traj]["D1"].append(np.nan)
                aligned_params[traj]["D2"].append(np.nan)
                aligned_params[traj]["theta"].append(np.nan)

    return aligned_params

def print_traj_params(trajectories):
    for traj in trajectories:
        D1, D2, theta = traj.D1, traj.D2, traj.theta

def print_traj_params_by_id(trajectories, id=None):
    for traj in trajectories:
        if traj.id == id:
            D1, D2, theta = traj.D1, traj.D2, traj.theta

            print(f"Parameters for trajectory ID {id}: D1={D1:.3f}, D2={D2:.3f}, theta={theta:.3f}")
            return

    print(f"No trajectory found with id {id}")

def print_traj_params_aligned(trajectories_list):
    aligned_params = get_traj_params_aligned(trajectories_list)

    print(f"Parameters (aligned by trajectory ID):")
    for traj, params in aligned_params.items():
        D1_values = ", ".join(f"{d:.3f}" if np.isfinite(d) else "nan" for d in params["D1"])
        D2_values = ", ".join(f"{d:.3f}" if np.isfinite(d) else "nan" for d in params["D2"])
        theta_values = ", ".join(f"{d:.3f}" if np.isfinite(d) else "nan" for d in params["theta"])
        print(f"Trajectory ID {traj.id}: D1=[{D1_values}], D2=[{D2_values}], theta=[{theta_values}]")

# =============================================================================
# Error metrics
# =============================================================================

def angular_error_pi_periodic(theta_est, theta_GT):
    """
    Angle error for diffusion axes, where theta and theta + pi are equivalent.
    Returns error in radians.
    """
    theta_est = np.asarray(theta_est, dtype=float)
    theta_GT = np.asarray(theta_GT, dtype=float)

    diff = np.abs(theta_est - theta_GT)
    diff = np.minimum(diff, np.pi - diff)

    return diff

def compute_tensor_MAE(
    trajectories_est,
    trajectories_GT,
    angle=False,
):
    """
    Compute MAE for D1 and D2, aligned by assigned trajectory IDs.
    """
    n_GT = len(trajectories_GT)

    D1_GT, D2_GT, theta_GT = get_traj_params(trajectories_GT)

    estimateDfromTrajectories(
        trajectories_est,
        dt=1.0,
        tau=1,
    )

    D1_est, D2_est, theta_est = get_traj_params(
        trajectories_est,
        n_GT=n_GT,
    )

    MAE_D1 = np.abs(D1_est - D1_GT)
    MAE_D2 = np.abs(D2_est - D2_GT)

    print(f"D1:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D1_est])

    print(f"D2:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D2_est])

    print(f"MAE D1:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in MAE_D1])

    print(f"MAE D2:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in MAE_D2])

    if angle:
        angle_error = angular_error_pi_periodic(theta_est, theta_GT)
        print("Angle error:")
        print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in angle_error])
        return MAE_D1, MAE_D2, angle_error

    return MAE_D1, MAE_D2

def compute_MAE(truth, estimate):
    if truth is None or estimate is None:
        return None
    return abs(truth - estimate)

def compute_D_HR(trajectories_HR):

    # compute and set MSD in trajectory objects
    calculateMSDtrajectories(trajectories_HR)
    
    # compute and print D_FR for each trajectory
    D_HR = estimateDfromTrajectories(trajectories_HR, 'HR')
    print(f"D_HR: \n[{', '.join(f'{float(d):.3f}' for d in D_HR)}]")
    
    return D_HR

def compute_MAE_HR(trajectories_HR, D_GT):

    D_HR = compute_D_HR(trajectories_HR)
    
    # compute and print MAE for D_trajectory
    MAE_HR = np.abs(np.array(D_HR) - np.array(D_GT))
    print(f"MAE for D_HR: \n[{', '.join(f'{float(d):.3f}' for d in MAE_HR)}]")  

    return MAE_HR

def compute_D_FR(trajectories_GT):

    # compute and set MSD in trajectory objects
    calculateMSDtrajectories(trajectories_GT)
    
    # compute and print D_FR for each trajectory
    D_FR = estimateDfromTrajectories(trajectories_GT, 'FR')
    print(f"D_FR: \n[{', '.join(f'{float(d):.3f}' for d in D_FR)}]")
    
    return D_FR

def compute_MAE_FR(trajectories_GT, D_GT):

    # compute and set MSD in trajectory objects
    calculateMSDtrajectories(trajectories_GT)
    
    # compute and print D_FR for each trajectory
    D_FR = estimateDfromTrajectories(trajectories_GT, 'FR')
    print(f"D_FR: \n[{', '.join(f'{float(d):.3f}' for d in D_FR)}]")
    
    # compute and print MAE for D_trajectory
    MAE_FR = np.abs(np.array(D_FR) - np.array(D_GT))
    print(f"MAE for D_FR: \n[{', '.join(f'{float(d):.3f}' for d in MAE_FR)}]")  

    return MAE_FR

def compute_D_detection(trajectories_detection, D_GT):
    calculateMSDtrajectories(trajectories_detection)

    estimateDfromTrajectories(trajectories_detection, 'detection')

    # initialize list aligned with GT
    D_detection = [np.nan] * len(D_GT)

    for traj in trajectories_detection:
        if traj.id is not None and 0 <= traj.id <= len(D_GT) - 1:
            D_detection[traj.id] = traj.D_detection

    print("D_detection:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D_detection])

    return D_detection

def compute_MAE_detection(trajectories_detection, D_GT):
    calculateMSDtrajectories(trajectories_detection)

    estimateDfromTrajectories(trajectories_detection, 'detection')

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

def compute_D_localization(trajectories_localization, D_GT):
    calculateMSDtrajectories(trajectories_localization)

    estimateDfromTrajectories(trajectories_localization, 'localization')

    D_localization = [np.nan] * len(D_GT)

    for traj in trajectories_localization:
        if traj.id is not None and 0 <= traj.id <= len(D_GT) - 1:
            D_localization[traj.id] = traj.D_localization

    print("D_localization:")
    print([f"{d:.3f}" if np.isfinite(d) else "nan" for d in D_localization])

    return D_localization

def compute_MAE_localization(trajectories_localization, D_GT):
    calculateMSDtrajectories(trajectories_localization)

    estimateDfromTrajectories(trajectories_localization, 'localization')

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

def evaluate_diffusion(
    trajectories_est,
    trajectories_GT,
    target="localization",
    dt=1.0,
    tau=1,
    angle=True,
):
    """
    Estimate diffusion tensors and compute errors against ground truth.

    Works for both isotropic and anisotropic diffusion. Isotropic diffusion is
    treated as the special case where D1 ≈ D2.

    Parameters
    ----------
    trajectories_est : list of Trajectory
        Estimated trajectories assigned to GT ids.
    trajectories_GT : list of Trajectory
        Ground-truth trajectories.
    target : {"trajectory", "detection", "localization"}
        Attribute namespace where estimated values are stored.
    dt : float
        Time between stored trajectory points.
    tau : int
        Lag used for tensor estimation.
    angle : bool
        If True, also compute angular error.

    Returns
    -------
    dict
        Dictionary containing GT parameters, estimated parameters, and errors.
    """
    n_GT = len(trajectories_GT)

    # estimate_diffusion_tensors(
    #     trajectories_est,
    #     dt=dt,
    #     tau=tau,
    #     target=target,
    # )

    D1_GT, D2_GT, theta_GT = get_traj_params(trajectories_GT, target='GT')

    D1_est, D2_est, theta_est = get_traj_params(
        trajectories_est,
        nparticles=n_GT,
        target=target
    )

    D_iso_GT = tensor_to_scalar_D(D1_GT, D2_GT)
    D_iso_est = tensor_to_scalar_D(D1_est, D2_est)

    errors = {
        "D1": np.abs(np.array(D1_est) - np.array(D1_GT)),
        "D2": np.abs(np.array(D2_est) - np.array(D2_GT)),
        "D_iso": np.abs(np.array(D_iso_est) - np.array(D_iso_GT))
    }

    if angle:
        errors["theta"] = angular_error_pi_periodic(theta_est, theta_GT)

    return {
        "target": target,
        "D1_GT": D1_GT,
        "D2_GT": D2_GT,
        "theta_GT": theta_GT,
        "D1_est": D1_est,
        "D2_est": D2_est,
        "theta_est": theta_est,
        "D_iso_GT": D_iso_GT,
        "D_iso_est": D_iso_est,
        "errors": errors,
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def plotMSDvsGT(trajectories_new, trajectories_GT):
    """
    Plot each detected trajectory MSD against the GT trajectory MSD
    matched by trajectory id.
    Plot only trajectories that were assigned a GT id (i.e. traj.id is not None) and for which the GT id exists in the GT trajectories list.

    Assumes:
        - each traj in trajectories_new has traj.id set to the matched GT id
        - each traj in trajectories_GT has its own GT id in traj.id
        - traj.MSD contains the MSD array/list
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

        MSD_new = traj_new.MSD
        gt_id = traj_new.id

        # Plot detected / experimental MSD
        if MSD_new is not None and len(MSD_new) > 0:
            plt.plot(MSD_new, label='Experimental MSD')

        # Plot matched GT MSD if id exists and is found
        if gt_id is not None and gt_id in gt_by_id:
            MSD_gt = gt_by_id[gt_id].MSD
            if MSD_gt is not None and len(MSD_gt) > 0:
                plt.plot(MSD_gt, ':', label='Ground-truth MSD')
            plt.title(f'New traj matched to GT ID {gt_id}')
        else:
            plt.title(f'New traj ID {gt_id} (No GT match)')

        plt.xlabel('Time lag (tau)')
        plt.ylabel('MSD')
        plt.legend()

    plt.tight_layout()
    plt.show()

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

