import numpy as np
import sys
from hungarian_algorithm import algorithm

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


def cog(trajectory):
    positions = trajectory.get_positions()
    if len(positions) == 0:
        print(f"Found no positions for trajectory {trajectory.id} -> set id to None")
        return None
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]
    cog_row = np.mean(rows) # simple mean in rows
    cog_col = np.mean(cols) # simple mean in cols
    return (float(cog_row), float(cog_col))

# cost function for assignment: distance between cogs
def cost_cog(traj_new, traj_GT):
    cog_new = cog(traj_new)
    cog_GT = cog(traj_GT)
    return np.linalg.norm(np.array(cog_new) - np.array(cog_GT))

# definition of the cost matrix
def compute_cost_matrix(trajectories_new, trajectories_GT):
    cost_matrix = np.zeros((len(trajectories_new), len(trajectories_GT)))
    for i, traj_new in enumerate(trajectories_new):
        for j, traj_GT in enumerate(trajectories_GT):
            cost_matrix[i, j] = cost_cog(traj_new, traj_GT)
    return cost_matrix

def relabel_from_assignment(trajectories_new, trajectories_GT, assignment):
    for i, traj in enumerate(trajectories_new):
        traj.id = assignment[i] # assign all trajectories according to the assignment; if no assignment was made, traj.id will be -1
        
        if traj.id == -1 or traj.id is None: # if no assignment was made, set id to -1 and color to gray
            traj.color = 'gray' # unassigned trajectories in gray
        else:
            traj.color = trajectories_GT[traj.id].color # assigned trajectories take the color of the GT trajectory they are assigned to
    return trajectories_new

def assign_trajectories(trajectories_new, trajectories_GT, algorithm='hungarian', verbose=False):
 
    cost = compute_cost_matrix(trajectories_new, trajectories_GT)

    if verbose:
        print("Cost matrix:")
        print(cost)

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
    else:
        raise ValueError("Invalid algorithm specified. Choose 'hungarian', 'hungarian_pypi', 'local_nn', or 'global_nn'.")

    if verbose:
        print_assignment(assignment)

    trajectories_new = relabel_from_assignment(trajectories_new, trajectories_GT, assignment)

    if verbose:
        for traj in trajectories_new:
            relabelled_count = 0
            if traj.id != -1 and traj.id is not None:
                relabelled_count += 1
            print(f"Trajectory ID: {traj.id}, Number of positions: {len(traj.get_positions())}, Relabelled: {'Yes' if traj.id != -1 and traj.id is not None else 'No'}")
    
    return trajectories_new, min_cost, assignment

def print_assignment(assignment):
    """"Print in the form of two columns with arrows in between, to visualize the assignment of trajectories."""
    for i, assigned_id in enumerate(assignment):
        print(f"{i} -> {assigned_id}")