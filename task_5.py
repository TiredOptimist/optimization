"""
REQUIRED LIBRARIES & FUNCTIONS (Resource Allocation DP)
====================================================================
1. numpy - for grid generation and DP tables (np.arange, np.zeros_like, np.searchsorted)
2. scipy.interpolate.interp1d - linear interpolation for future costs (kind='linear')
   (Alternative: manual linear interpolation is also implemented, but interp1d is used here
    to match the method described in the manual.)

Additional libraries from the general list (linprog, minimize_scalar, deque) are NOT
needed for this DP problem, as we use discrete grid search and linear interpolation.

This script solves the resource allocation problem with:
Z = 16, N = 3,
g(Y) = 4Y + 0.22Y^2,
h(X-Y) = 5(X-Y) + 0.14(X-Y)^2,
alpha = 0.7, beta = 0.3,
state grid = {0, 4, 8, 12, 16}, control step = 2.

It computes Bellman tables F_k(X) and Y_k(X) for k=1..N,
then recovers the optimal trajectory for X0 = 16.
"""

import numpy as np
from scipy.interpolate import interp1d

# Problem parameters
Z = 16                      # initial resource
N = 3                       # number of stages
ALPHA = 0.7                 # retention coefficient for sphere A
BETA = 0.3                  # retention coefficient for sphere B

# Cost functions
def g(Y: float) -> float:
    """Cost of investing Y in sphere A: g(Y) = 4Y + 0.22Y^2."""
    return 4.0 * Y + 0.22 * Y * Y

def h(X_minus_Y: float) -> float:
    """Cost of investing (X - Y) in sphere B: h = 5*(X-Y) + 0.14*(X-Y)^2."""
    return 5.0 * X_minus_Y + 0.14 * X_minus_Y * X_minus_Y

def stage_cost(Y: float, X: float) -> float:
    """Immediate cost for a given control Y and current resource X."""
    return g(Y) + h(X - Y)

# Discretization
X_grid = np.array([0, 4, 8, 12, 16], dtype=float)   # state grid
Y_STEP = 2.0                                        # control search step


# DP solver
def solve_dp() -> tuple:
    """
    Solves the DP problem for all grid points.

    Returns:
        X_grid (np.ndarray): state grid
        F_tables (list of np.ndarray): F_k(X) for k=1..N, each array aligned with X_grid
        Y_tables (list of np.ndarray): Y_k(X) for k=1..N, each array aligned with X_grid
    """
    # Initialize storage
    F_tables = []   # F_tables[k-1][i] = F_k(X_grid[i])
    Y_tables = []   # Y_tables[k-1][i] = Y_k(X_grid[i])

    # k = 1: base case 
    F1 = np.zeros_like(X_grid)
    Y1 = np.zeros_like(X_grid)

    for i, X in enumerate(X_grid):
        if X == 0:
            F1[i] = 0.0
            Y1[i] = 0.0
            continue

        best_val = float('inf')
        best_Y = 0.0
        # Generate control candidates from 0 to X with step Y_STEP
        Y_candidates = np.arange(0.0, X + Y_STEP, Y_STEP)
        for Y in Y_candidates:
            cost = stage_cost(Y, X)
            if cost < best_val:
                best_val = cost
                best_Y = Y
        F1[i] = best_val
        Y1[i] = best_Y

    F_tables.append(F1)
    Y_tables.append(Y1)

    # k = 2..N: recursion with interpolation 
    for k in range(2, N + 1):
        Fk = np.zeros_like(X_grid)
        Yk = np.zeros_like(X_grid)
        F_prev = F_tables[-1]   # F_{k-1}

        # Create an interpolation function for F_prev (linear)
        # This is more convenient and matches the manual's approach.
        interp_future = interp1d(X_grid, F_prev, kind='linear', fill_value='extrapolate')

        for i, X in enumerate(X_grid):
            if X == 0:
                Fk[i] = 0.0
                Yk[i] = 0.0
                continue

            best_val = float('inf')
            best_Y = 0.0
            Y_candidates = np.arange(0.0, X + Y_STEP, Y_STEP)
            for Y in Y_candidates:
                immediate = stage_cost(Y, X)
                # Next state resource
                X_next = ALPHA * Y + BETA * (X - Y)
                # Interpolate future cost from F_prev using interp1d
                future = float(interp_future(X_next))  # interp1d returns numpy scalar
                total = immediate + future
                if total < best_val:
                    best_val = total
                    best_Y = Y
            Fk[i] = best_val
            Yk[i] = best_Y

        F_tables.append(Fk)
        Y_tables.append(Yk)

    return X_grid, F_tables, Y_tables


# Trajectory recovery 
def recover_trajectory(X_initial: float,
                       X_grid: np.ndarray,
                       Y_tables: list) -> list:
    """
    Reconstructs the optimal sequence of controls Y_1, Y_2, ..., Y_N
    for a given initial resource X_initial using the precomputed Y_k tables.
    Linear interpolation (via interp1d) is used for states that do not lie exactly on the grid.

    Returns:
        list of float: optimal controls in forward order [y1, y2, ..., yN]
    """
    trajectory_rev = []   # will store y_N, y_{N-1}, ..., y_1
    X_current = X_initial

    # Process stages from N down to 1
    for k in range(N, 0, -1):
        Yk = Y_tables[k - 1]   # table Y_k
        # Create interpolation for Y_k (linear)
        interp_Yk = interp1d(X_grid, Yk, kind='linear', fill_value='extrapolate')
        # Interpolate optimal control for current state X_current
        Y_opt = float(interp_Yk(X_current))
        trajectory_rev.append(Y_opt)
        # Update resource for the next (previous) stage
        X_current = ALPHA * Y_opt + BETA * (X_current - Y_opt)

    # Reverse to obtain forward order: [y1, y2, ..., yN]
    return trajectory_rev

# Main execution
def main():
    """Main function: solve DP, print tables and optimal trajectory."""
    print("Numerical DP Solver for Resource Allocation\n")

    # Solve DP
    X_grid, F_tables, Y_tables = solve_dp()

    # Print table for all grid points
    print(f"{'X':>3}  {'F1':>8}  {'Y1':>8}  {'F2':>8}  {'Y2':>8}  {'F3':>8}  {'Y3':>8}")
    print("-" * 60)
    for i, X in enumerate(X_grid):
        f1, y1 = F_tables[0][i], Y_tables[0][i]
        f2, y2 = F_tables[1][i], Y_tables[1][i]
        f3, y3 = F_tables[2][i], Y_tables[2][i]
        print(f"{X:3.0f}  {f1:8.2f}  {y1:8.2f}  {f2:8.2f}  {y2:8.2f}  {f3:8.2f}  {y3:8.2f}")

    # Recover trajectory for X0 = Z
    print(f"\nOptimal trajectory for X0 = {Z}, N = {N}")
    traj = recover_trajectory(float(Z), X_grid, Y_tables)
    for step, y_val in enumerate(traj, start=1):
        print(f"Step {step}: y{step} = {y_val:.2f}")

    # Total minimal cost
    total_cost = F_tables[N - 1][np.where(X_grid == Z)[0][0]]
    print(f"\nMinimum total cost F_{N}({Z}) = {total_cost:.2f}")


if __name__ == "__main__":
    main()