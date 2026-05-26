import numpy as np
from hungarian_algorithm import algorithm

DEFAULT_COST_DISTANCE_NORM = 8.0
DEFAULT_COST_INTENSITY_NORM = 700.0
DEFAULT_COST_SIGMA_NORM = 0.75
DEFAULT_COST_LENGTH_NORM = 10.0
DEFAULT_COST_START_FRAME_NORM = 5.0


def resolve_cost_norm(value, fallback):
    if value is None:
        return float(fallback)

    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return float(fallback)

    if values.shape == ():
        norm = float(values)
        return norm if np.isfinite(norm) and norm > 0 else float(fallback)

    positive_values = values[np.isfinite(values) & (values > 0)]
    if positive_values.size == 0:
        return float(fallback)

    return float(np.max(positive_values))


# ------------------------ Assignment algorithms -------------------------
def local_nn_assignment(cost_matrix, max_distance=10):
    """
    Local nearest neighbor assignment based on the cost matrix. Can be used as a simple baseline for assignment, but does 
    not guarantee a globally optimal solution and can lead to conflicts (multiple rows assigned to the same column).
    Args:
        cost_matrix: a 2D numpy array where cost_matrix[i][j] represents the cost of assigning trajectory i to trajectory j.
        max_distance: maximum allowed distance for assignment; if the minimum cost exceeds this threshold, no assignment is made.
    Returns:
        min_cost: the minimum cost of the matching
        assignment: a list where assignment[i] is the index of the trajectory assigned to trajectory i, or -1 if no assignment is made.
    """
    assignment = [-1] * len(cost_matrix) # initialize assignment with -1
    for i in range(len(cost_matrix)):
        min_cost = np.min(cost_matrix[i]) # find minimum cost for row i
        min_index = np.argmin(cost_matrix[i]) # find index of minimum cost
        if min_cost <= max_distance: # if minimum cost is within threshold, assign
            assignment[i] = min_index
    min_cost = 0
    for i in range(len(cost_matrix)):
        if assignment[i] != -1: # if there is an assignment for row i
            min_cost += cost_matrix[i][assignment[i]] # sum the cost of the assigned pairs
    return min_cost, assignment

def global_nn_assignment(cost_matrix, max_distance=10):
    """
    Global nearest neighbor assignment based on the cost matrix. Principle: first get the local NN assignment, then check for conflicts 
    (multiple rows assigned to the same column) and resolve them by keeping only the assignment with the lowest cost for that column.
    Args:
        cost_matrix: a 2D numpy array where cost_matrix[i][j] represents the cost of assigning trajectory i to trajectory j.
        max_distance: maximum allowed distance for assignment; if the minimum cost exceeds this threshold, no assignment is made.
    Returns:
        min_cost: the minimum cost of the matching
        assignment: a list where assignment[i] is the index of the trajectory assigned to trajectory i, or -1 if no assignment is made.
    """
    min_cost, assignment = local_nn_assignment(cost_matrix, max_distance) # get local NN assignment
    for i in range(len(assignment)):
        if assignment[i] != -1: # if there is an assignment for row i
            j = assignment[i]
            # check if this column j is assigned to multiple rows
            assigned_rows = [k for k in range(len(assignment)) if assignment[k] == j]
            if len(assigned_rows) > 1: # if multiple rows are assigned to the same column
                # find the row with the lowest cost for this column
                costs_for_j = [cost_matrix[k][j] for k in assigned_rows]
                min_cost_index = np.argmin(costs_for_j)
                best_row = assigned_rows[min_cost_index]
                # unassign all other rows except the best one
                for k in assigned_rows:
                    if k != best_row:
                        assignment[k] = -1
    min_cost = 0
    for i in range(len(cost_matrix)):
        if assignment[i] != -1:
            min_cost += cost_matrix[i][assignment[i]]
    return min_cost, assignment

def greedy_one_to_one_assignment(cost_matrix, max_distance=10):
    """Greedy one-to-one assignment that repeatedly selects the cheapest valid row-column pair.

    This is a strict one-to-one baseline: each row and each column can be used at most once.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    n_rows, n_cols = cost_matrix.shape
    assignment = np.full(n_rows, -1, dtype=int)

    if n_rows == 0 or n_cols == 0:
        return 0.0, assignment.tolist()

    candidate_pairs = []
    for i in range(n_rows):
        for j in range(n_cols):
            cost = float(cost_matrix[i, j])
            if not np.isfinite(cost):
                continue
            if max_distance is not None and cost > max_distance:
                continue
            candidate_pairs.append((cost, i, j))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    used_rows = set()
    used_cols = set()
    min_cost = 0.0

    for cost, i, j in candidate_pairs:
        if i in used_rows or j in used_cols:
            continue
        assignment[i] = j
        used_rows.add(i)
        used_cols.add(j)
        min_cost += cost

    return float(min_cost), assignment.tolist()

def mutual_nn_assignment(cost_matrix, max_distance=10):
    """Mutual nearest-neighbor assignment with one-to-one matching.

    At each round, rows and columns that are mutually closest are matched and removed.
    This is stricter than local NN and cheaper than Hungarian, but it is still heuristic.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    n_rows, n_cols = cost_matrix.shape
    assignment = np.full(n_rows, -1, dtype=int)

    if n_rows == 0 or n_cols == 0:
        return 0.0, assignment.tolist()

    remaining_rows = list(range(n_rows))
    remaining_cols = list(range(n_cols))
    min_cost = 0.0

    while remaining_rows and remaining_cols:
        row_best = {}
        col_best = {}

        for i in remaining_rows:
            row_values = cost_matrix[i, remaining_cols]
            if row_values.size == 0:
                continue
            best_pos = int(np.argmin(row_values))
            best_col = remaining_cols[best_pos]
            best_cost = float(row_values[best_pos])
            if np.isfinite(best_cost) and (max_distance is None or best_cost <= max_distance):
                row_best[i] = (best_col, best_cost)

        for j in remaining_cols:
            col_values = cost_matrix[remaining_rows, j]
            if col_values.size == 0:
                continue
            best_pos = int(np.argmin(col_values))
            best_row = remaining_rows[best_pos]
            best_cost = float(col_values[best_pos])
            if np.isfinite(best_cost) and (max_distance is None or best_cost <= max_distance):
                col_best[j] = (best_row, best_cost)

        mutual_pairs = []
        for i, (j, cost) in row_best.items():
            best_row_for_col = col_best.get(j)
            if best_row_for_col is not None and best_row_for_col[0] == i:
                mutual_pairs.append((cost, i, j))

        if not mutual_pairs:
            break

        mutual_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        used_rows = set()
        used_cols = set()

        for cost, i, j in mutual_pairs:
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            assignment[i] = j
            min_cost += cost

        if not used_rows:
            break

        remaining_rows = [i for i in remaining_rows if i not in used_rows]
        remaining_cols = [j for j in remaining_cols if j not in used_cols]

    return float(min_cost), assignment.tolist()

def hungarian(cost):
    """Hungarian algorithm to find the minimum cost matching in a weighted bipartite graph.
    Inspired by https://every-algorithm.github.io/2024/08/27/hungarian_algorithm.html
    Args:
        cost: a 2D numpy array where cost[i][j] represents the cost of assigning trajectory i to trajectory j.
    Returns:
        min_cost: the minimum cost of the matching
        assignment: a list where assignment[i] is the index of the trajectory assigned to trajectory i, or -1 if no assignment is made. 
        No assignemnt for trajectory i is pointed out by assignment[i] = -1.
    """
    
    n = len(cost)
    m = len(cost[0])
    size = max(n, m) # rectangular cost matrix if diff num of trajectories on one frame and the next

    dummy_cost = 1e6 # a large cost for dummy assignments

    # pad to square matrix
    padded_cost = np.full((size, size), dummy_cost)
    for i in range(n):
        for j in range(m):
            padded_cost[i][j] = cost[i][j]

    u = np.zeros(size) # potential for left vertices
    v = np.zeros(size) # potential for right vertices
    p = np.zeros(size + 1, dtype=int) # p[j] = i means j is matched to i, p[0] is dummy
    way = np.zeros(size + 1, dtype=int) # way[j] = k means j is connected to k in the alternating path

    # Hungarian algorithm
    for i in range(1, size + 1):
        p[0] = i # start with unmatched left vertex i
        minv = np.full(size + 1, np.inf) # minv[j] = minimum reduced cost to connect j to the alternating tree
        used = np.zeros(size + 1, dtype=bool) # used[j] = True means j is in the alternating tree
        j0 = 0 # current column to add to the tree

        # find minimum cost matching for row i
        while True:
            # add j0 to the tree
            used[j0] = True # mark j0 as used
            i0 = p[j0] # i0 is the row currently matched to j0
            j1 = 0 # j1 will be the next column to consider
            delta = np.inf # minimum reduced cost to add a new edge
            # iterate over all columns to find the one with the smallest reduced cost to connect to the tree
            for j in range(1, size + 1):
                if not used[j]: # if j is not used, consider adding edge (i0, j)
                    cur = padded_cost[i0 - 1][j - 1] - u[i0 - 1] - v[j - 1] # reduced cost of edge (i0, j)
                    if cur < minv[j]: # if this edge has smaller reduced cost, update minv and way
                        minv[j] = cur # update minimum reduced cost for j
                        way[j] = j0 # update the way to connect j to the tree
                    if minv[j] < delta: # update delta and j1 if this is the smallest reduced cost found
                        delta = minv[j] # new minimum reduced cost
                        j1 = j # new column with the smallest reduced cost
            for j in range(size + 1): # update potentials
                if used[j]: #  if j is in the tree, update u and v
                    if p[j] != 0:
                        u[p[j] - 1] += delta # increase potential of matched row
                    if j != 0:
                        v[j - 1] -= delta # decrease potential of column
                else:
                    minv[j] -= delta # decrease minimum reduced cost for unmatched columns
            j0 = j1 # add j1 to the tree
            if p[j0] == 0:
                break
        while True: # augment along the path
            j1 = way[j0] # j1 is the column that connects j0 to the tree
            p[j0] = p[j1] # match j0 to the same row as j1
            j0 = j1 # move to the next column in the path
            if j0 == 0:
                break

    # extract the matching from p
    assignment = np.full(n, -1, dtype=int) # assignment[j] = i means j is matched to i
    for j in range(1, size + 1):
        if p[j] <= n and j <= m and p[j] != 0: # only consider valid matches within original matrix size
            assignment[p[j] - 1] = j - 1 # assign row p[j] to column j; -1 for 0-based indexing
    min_cost = 0 # calculate the minimum cost of the matching
    for i in range(n):
        if assignment[i] != -1: # if row i is assigned to a column
            min_cost += cost[i][assignment[i]] # sum the cost of the assigned pairs; assignment[i] gives the column assigned to row i

    return min_cost, assignment 

def hungarian_pypi(cost):
    """
    Hungarian algorithm using the 'hungarian_algorithm' package from PyPI,
    to find the minimum cost matching in a weighted bipartite graph.

    Args:
        cost: a weighted bipartite graph (list of lists or numpy array)

    Returns:
        min_cost: the minimum cost of the matching
        assignment: list where assignment[i] = j means row i is matched to column j
    """
    # convert cost to list of lists
    if isinstance(cost, np.ndarray):
        cost = cost.tolist()

    n_rows = len(cost)
    n_cols = len(cost[0])

    # pad cost to square matrix with infinite dummy cost
    size = max(n_rows, n_cols)
    padded_cost = np.full((size, size), np.inf)
    for i in range(n_rows):
        for j in range(n_cols):
            padded_cost[i][j] = cost[i][j]

    # convert to graph format expected by the package
    graph = {
        f"r{i}": {f"c{j}": padded_cost[i][j] for j in range(size)}
        for i in range(size)
    }

    result = algorithm.find_matching(
        graph,
        matching_type='min',
        return_type='list'
    )  # format: [((row_label, col_label), weight), ...]

    # convert result to assignment format
    assignment = [-1] * n_rows
    min_cost = 0.0

    for (row_node, col_node), weight in result:
        # extract indices
        if row_node.startswith("r") and col_node.startswith("c"):
            i = int(row_node[1:])
            j = int(col_node[1:])
        else:
            continue

        # ignore dummy padded rows/cols
        if i < n_rows and j < n_cols:
            assignment[i] = j
            min_cost += weight

    return min_cost, assignment

# ------------------------- Cost functions and cost matrix for assignment -------------------------
def cog(trajectory):
    positions = trajectory.get_positions()
    if len(positions) == 0:
        return None

    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]
    cog_row = np.mean(rows)
    cog_col = np.mean(cols)
    return (float(cog_row), float(cog_col))

# cost function for assignment: distance between cogs
def cost_cog(traj_new, traj_GT, invalid_cost=np.inf):
    cog_new = cog(traj_new)
    cog_GT = cog(traj_GT)

    if cog_new is None or cog_GT is None:
        return invalid_cost

    return np.linalg.norm(np.array(cog_new) - np.array(cog_GT))


# definition of the cost matrix
def compute_cost_matrix_trajectories(trajectories_new, trajectories_GT, cost_function=None):
    if cost_function is None:
        cost_function = TrajToTraj.default()

    cost_matrix = np.zeros((len(trajectories_new), len(trajectories_GT)))
    for i, traj_new in enumerate(trajectories_new):
        for j, traj_GT in enumerate(trajectories_GT):
            cost_value = cost_function(traj_new, traj_GT)

            if isinstance(cost_value, tuple):
                cost_value = cost_value[0]

            if torch.is_tensor(cost_value):
                cost_value = cost_value.detach().cpu().item()

            cost_matrix[i, j] = float(cost_value)
    return cost_matrix

def compute_cost_matrix_p2p(peaks_f1, peaks_f2, distance=True, intensity_diff=False, sigma_diff=False):
    cost_matrix = np.zeros((len(peaks_f1), len(peaks_f2)))
    for i, peak1 in enumerate(peaks_f1):
        for j, peak2 in enumerate(peaks_f2):
            cost = 0
            if distance:
                cost += np.linalg.norm(np.array(peak1[:2]) - np.array(peak2[:2]))
            if intensity_diff:
                # TO DO
                continue
            if sigma_diff:
                # TO DO
                continue
            cost_matrix[i, j] = cost
    return cost_matrix

def compute_cost_matrix_tracks_to_detections(
    active_trajectories,
    current_detections,
    cost_function=None,
    max_distance=None,
    invalid_cost=1e6,
):
    """
    Rows = active trajectories
    Cols = current detections

    Each detection is expected to be normalized as:

        (pos, intensity, sigma)

    where:

        pos = (x, y) or (x, y, z)

    Spatial gating is always done using raw Euclidean distance.

    `cost_function` should be callable:

        cost = cost_function(trajectory, detection)
    """

    if cost_function is None:
        cost_function = PeakToPeak.default()

    cost_matrix = np.full(
        (len(active_trajectories), len(current_detections)),
        invalid_cost,
        dtype=float,
    )

    for i, traj in enumerate(active_trajectories):
        last_pos = np.asarray(traj.last_position(), dtype=float)
        if max_distance is None:
            row_max_distance = None
        elif np.isscalar(max_distance):
            row_max_distance = float(max_distance)
        else:
            row_max_distance = float(max_distance[i])

        for j, det in enumerate(current_detections):
            pos = np.asarray(det[0], dtype=float)

            spatial_dist = np.linalg.norm(last_pos - pos)

            if row_max_distance is not None and spatial_dist > row_max_distance:
                continue

            cost = cost_function(traj, det)

            if isinstance(cost, tuple):
                cost = cost[0]

            if torch.is_tensor(cost):
                cost = cost.detach().cpu().item()

            cost_matrix[i, j] = float(cost)

    return cost_matrix


# ------------------------- Check for temporal overlap of trajectories -------------------------
# check if two trajectories overlap in time
def trajectories_overlap_in_time(traj1, traj2): 
    return not (traj1.end_frame < traj2.start_frame or traj2.end_frame < traj1.start_frame)

# check the interval of frames where two trajectories overlap in time; return None if they do not overlap
def overlap_interval(traj1, traj2): # 
    start = max(traj1.start_frame, traj2.start_frame)
    end = min(traj1.end_frame, traj2.end_frame)
    if start > end:
        return None
    return start, end

# cost function that only considers the distance between cogs if the trajectories overlap in time, otherwise returns a large cost
def cost_cog_overlap_only(traj_new, traj_GT, invalid_cost=1e6):
    if not trajectories_overlap_in_time(traj_new, traj_GT):
        return invalid_cost
    return cost_cog(traj_new, traj_GT, invalid_cost=invalid_cost)

# cost function that computes the mean distance between positions of two trajectories on their temporal overlap; if no overlap, returns a large cost
def mean_position_distance_on_overlap(traj1, traj2, invalid_cost=1e6): 
    interval = overlap_interval(traj1, traj2)
    if interval is None:
        return invalid_cost

    start, end = interval
    distances = []

    for f in range(start, end + 1):
        p1 = traj1.get_position_at_frame(f)
        p2 = traj2.get_position_at_frame(f)

        if p1 is None or p2 is None:
            continue

        distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))

    if len(distances) == 0:
        return invalid_cost

    return float(np.mean(distances))

# ------------------------- Relabel trajectories according to assignment -------------------------
def relabel_from_assignment(trajectories_new, trajectories_GT, assignment):
    for i, traj in enumerate(trajectories_new):
        traj.id = assignment[i] # assign all trajectories according to the assignment; if no assignment was made, traj.id will be -1
        
        if traj.id == -1 or traj.id is None: # if no assignment was made, set id to -1 and color to gray
            traj.color = 'gray' # unassigned trajectories in gray
        else:
            traj.color = trajectories_GT[traj.id].color # assigned trajectories take the color of the GT trajectory they are assigned to
    return trajectories_new

# def assign_trajectories(trajectories_new, trajectories_GT, algorithm='hungarian', verbose=False):
 
#     cost = compute_cost_matrix(trajectories_new, trajectories_GT)

#     if verbose:
#         print("Cost matrix:")
#         print(cost)

#     if verbose:
#         print('Computing assignment using algorithm:', algorithm)
#     if algorithm == 'hungarian':
#         min_cost, assignment = hungarian(cost)
#     elif algorithm == 'hungarian_pypi':
#         min_cost, assignment = hungarian_pypi(cost)
#     elif algorithm == 'local_nn':
#         min_cost, assignment = local_nn_assignment(cost)
#     elif algorithm == 'global_nn':
#         min_cost, assignment = global_nn_assignment(cost)
#     else:
#         raise ValueError("Invalid algorithm specified. Choose 'hungarian', 'hungarian_pypi', 'local_nn', or 'global_nn'.")

#     if verbose:
#         print_assignment(assignment)

#     trajectories_new = relabel_from_assignment(trajectories_new, trajectories_GT, assignment)

#     if verbose:
#         for traj in trajectories_new:
#             relabelled_count = 0
#             if traj.id != -1 and traj.id is not None:
#                 relabelled_count += 1
#             print(f"Trajectory ID: {traj.id}, Number of positions: {len(traj.get_positions())}, Relabelled: {'Yes' if traj.id != -1 and traj.id is not None else 'No'}")
    
#     return trajectories_new, min_cost, assignment

def assign_trajectories(
    trajectories_new,
    trajectories_GT,
    algorithm='hungarian',
    cost_function=None,
    verbose=False
):
    """Assign trajectories from trajectories_new to trajectories_GT based on the specified algorithm and cost function.
    Args:
         trajectories_new: list of trajectories to be assigned
         trajectories_GT: list of ground truth trajectories to assign to
         algorithm: assignment algorithm to use ('hungarian', 'hungarian_pypi', 'local_nn', 'global_nn', 'greedy_nn', 'mutual_nn')
         cost_function: cost function to compute the cost matrix; if None, uses the recommended TrajToTraj default
         verbose: if True, print detailed information about the cost matrix and assignment process
    Returns:
         trajectories_new: list of trajectories with updated IDs according to the assignment
         min_cost: the minimum cost of the assignment
         assignment: list where assignment[i] is the index of the trajectory in trajectories_GT assigned to trajectories_new[i], or -1 if no assignment was made
    """
    if len(trajectories_new) == 0:
        if verbose:
            print("No new trajectories to assign.")
        return trajectories_new, np.inf, []

    if len(trajectories_GT) == 0:
        if verbose:
            print("No ground-truth trajectories available.")
        return trajectories_new, np.inf, [-1] * len(trajectories_new)

    if cost_function is None:
        cost_function = TrajToTraj.default()
    cost = compute_cost_matrix_trajectories(trajectories_new, trajectories_GT, cost_function=cost_function)

    if verbose:
        print('Computing assignment using algorithm:', algorithm)

    if algorithm == 'hungarian':
        min_cost, assignment = hungarian(cost)
    elif algorithm == 'hungarian_pypi':
        min_cost, assignment = hungarian_pypi(cost)
    elif algorithm == 'local_nn':
        min_cost, assignment = local_nn_assignment(cost)
    elif algorithm == 'global_nn':
        min_cost, assignment = global_nn_assignment(cost)
    elif algorithm == 'greedy_nn':
        min_cost, assignment = greedy_one_to_one_assignment(cost)
    elif algorithm == 'mutual_nn':
        min_cost, assignment = mutual_nn_assignment(cost)
    else:
        raise ValueError("Invalid algorithm specified. Choose 'hungarian', 'hungarian_pypi', 'local_nn', 'global_nn', 'greedy_nn', or 'mutual_nn'.")

    if verbose:
        print_assignment(assignment)

    trajectories_new = relabel_from_assignment(trajectories_new, trajectories_GT, assignment)

    if verbose:
        relabelled_count = 0
        for traj in trajectories_new:
            was_relabelled = traj.id != -1 and traj.id is not None
            if was_relabelled:
                relabelled_count += 1
            print(
                f"Trajectory ID: {traj.id}, "
                f"Frames: {traj.start_frame}->{traj.end_frame}, "
                f"Number of positions: {len(traj.get_positions())}, "
                f"Relabelled: {'Yes' if was_relabelled else 'No'}"
            )

    return trajectories_new, min_cost, assignment

def print_assignment(assignment):
    """"Print in the form of two columns with arrows in between, to visualize the assignment of trajectories."""
    for i, assigned_id in enumerate(assignment):
        print(f"{i} -> {assigned_id}")

def print_assignment_verbose(trajectories_new, trajectories_GT, assignment):
    for i, assigned_id in enumerate(assignment):
        new_traj = trajectories_new[i]
        if assigned_id == -1:
            print(
                f"new[{i}] ({new_traj.start_frame}->{new_traj.end_frame}) -> unassigned"
            )
        else:
            gt_traj = trajectories_GT[assigned_id]
            print(
                f"new[{i}] ({new_traj.start_frame}->{new_traj.end_frame}) "
                f"-> gt[{assigned_id}] ({gt_traj.start_frame}->{gt_traj.end_frame})"
            )
    
def assign_d_l_trajectories(trajectories_detection, trajectories_localization, trajectories_GT, algorithm='hungarian', verbose=False):
    # assignment
    traj_det, cost_det, assignment_det = assign_trajectories(trajectories_detection, trajectories_GT, algorithm=algorithm)
    traj_loc, cost_loc, assignment_loc = assign_trajectories(trajectories_localization, trajectories_GT)

    if verbose:
        print("Detection assignment:")
        print_assignment_verbose(traj_det, trajectories_GT, assignment_det)
        print("\nLocalization assignment:")
        print_assignment_verbose(traj_loc, trajectories_GT, assignment_loc)

    return traj_det, cost_det, assignment_det, traj_loc, cost_loc, assignment_loc


    # ---- OLD ------
def label_trajectories_from_GT(trajectories_new, trajectories_GT, max_distance=10):
    used_GT_idx = []

    for traj in trajectories_new:
        cog_traj = cog(traj)
        if cog_traj is None:
            traj.id = None
            continue

        distances = []
        for gt_traj in trajectories_GT:
            if not trajectories_overlap_in_time(traj, gt_traj): # if trajectories do not overlap in time, they cannot correspond to the same particle
                continue

            cog_gt = cog(gt_traj)
            if cog_gt is None:
                distances.append(np.inf)
                continue

            distances.append(np.linalg.norm(np.array(cog_traj) - np.array(cog_gt)))

        for idx in used_GT_idx:
            distances[idx] = np.inf

        closest_GT_idx = np.argmin(distances)
        if distances[closest_GT_idx] > max_distance:
            traj.set_id(None)
            print('No GT trajectory within max distance for', cog_traj, '-> id None')
        else:
            traj.set_id(trajectories_GT[closest_GT_idx].id)
            traj.color = trajectories_GT[closest_GT_idx].color
            used_GT_idx.append(closest_GT_idx)
            print('Assigned', cog_traj, 'to GT id', traj.id)

# ------- DEFINE CUSTOM COST FUNCTIONS FOR PARTICLE-TO-PARTICLE ASSIGNMENT AND TRAJECTORY-TO-TRAJECTORY ASSIGNMENT -------
import torch
import torch.nn as nn


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
    def default(cls, distance_norm=None, intensity_norm=None, sigma_norm=None):
        """Recommended peak-to-peak cost from CostExperiment."""
        distance_norm = resolve_cost_norm(distance_norm, DEFAULT_COST_DISTANCE_NORM)
        intensity_norm = resolve_cost_norm(intensity_norm, DEFAULT_COST_INTENSITY_NORM)
        sigma_norm = resolve_cost_norm(sigma_norm, DEFAULT_COST_SIGMA_NORM)

        return cls(
            terms={
                "distance": DistanceTerm(weight=0.5, norm=distance_norm),
                "intensity": IntensityTerm(weight=0.4, norm=intensity_norm),
                "sigma": SigmaTerm(weight=0.1, norm=sigma_norm),
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
    
class TrajToTrajTerm(nn.Module):
    def __init__(self, weight=1.0, norm=1.0, enabled=True):
        super().__init__()
        self.weight = weight
        self.norm = norm
        self.enabled = enabled

    def forward(self, traj1, traj2):
        raise NotImplementedError

class MeanPositionDistanceTerm(TrajToTrajTerm):
    def __init__(self, weight=1.0, norm=1.0, enabled=True, invalid_cost=1e6):
        super().__init__(weight=weight, norm=norm, enabled=enabled)
        self.invalid_cost = invalid_cost

    def forward(self, traj1, traj2):
        if not self.enabled:
            return torch.tensor(0.0)

        interval = overlap_interval(traj1, traj2)

        if interval is None:
            return torch.tensor(self.invalid_cost)

        start, end = interval
        distances = []

        for f in range(start, end + 1):
            p1 = traj1.get_position_at_frame(f)
            p2 = traj2.get_position_at_frame(f)

            if p1 is None or p2 is None:
                continue

            p1 = torch.as_tensor(p1, dtype=torch.float32)
            p2 = torch.as_tensor(p2, dtype=torch.float32)

            distances.append(torch.linalg.norm(p1 - p2))

        if len(distances) == 0:
            return torch.tensor(self.invalid_cost)

        mean_dist = torch.mean(torch.stack(distances))

        return self.weight * (mean_dist / self.norm)
    
class TrajToTraj(nn.Module):
    def __init__(self, terms=None, return_breakdown=False):
        super().__init__()

        if terms is None:
            terms = {}

        self.terms = nn.ModuleDict(terms)
        self.return_breakdown = return_breakdown

    @classmethod
    def default(cls, position_norm=None, length_norm=None):
        """Recommended trajectory-to-trajectory cost from CostExperiment."""
        position_norm = resolve_cost_norm(position_norm, DEFAULT_COST_DISTANCE_NORM)
        length_norm = resolve_cost_norm(length_norm, DEFAULT_COST_LENGTH_NORM)

        return cls(
            terms={
                "position": MeanPositionDistanceTerm(
                    weight=0.75,
                    norm=position_norm,
                ),
                "length": LengthDifferenceTerm(
                    weight=0.25,
                    norm=length_norm,
                ),
            }
        )

    def forward(self, traj1, traj2):
        costs = {
            name: term(traj1, traj2)
            for name, term in self.terms.items()
        }

        total = sum(costs.values())

        if self.return_breakdown:
            return total, costs

        return total
    
class LengthDifferenceTerm(TrajToTrajTerm):
    def forward(self, traj1, traj2):
        if not self.enabled:
            return torch.tensor(0.0)

        diff = abs(traj1.length() - traj2.length())
        return torch.tensor(self.weight * (diff / self.norm), dtype=torch.float32)
    
class StartFrameDifferenceTerm(TrajToTrajTerm):
    def forward(self, traj1, traj2):
        if not self.enabled:
            return torch.tensor(0.0)

        diff = abs(traj1.start_frame - traj2.start_frame)
        return torch.tensor(self.weight * (diff / self.norm), dtype=torch.float32)
