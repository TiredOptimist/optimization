"""
Numerical Dynamic Programming for Nonlinear Problem
Variant: Minimize total cost over 4 stages, initial resource X=14,
         immediate cost: 4Y^2 + 3(X-Y)^2,
         resource transition: X_next = 0.35Y + 0.65(X-Y)

Required libraries:
- numpy: grid generation, array storage, searchsorted for interpolation
- typing: type hints (List, Tuple)
"""

import numpy as np
from typing import List, Tuple

# Problem parameters
ALPHA = 0.35         # retention coefficient for part Y
BETA = 0.65          # retention coefficient for part (X-Y)
N_STEPS = 4          # number of stages (k = 1..4)
X_MAX = 14           # maximum initial resource
GRID_STEP = 1.0      # discretization step for state X
Y_STEP = 0.01        # search step for control Y

def stage_cost(Y: float, X: float) -> float:
    """
    Immediate cost for one stage: 4*Y^2 + 3*(X-Y)^2
    """
    return 4.0 * Y * Y + 3.0 * (X - Y) * (X - Y)

def linear_interpolate(x_vals: np.ndarray, y_vals: np.ndarray, x_query: float) -> float:
    """
    Linear interpolation of function y = f(x) given discrete points.
    Used to estimate F_{k-1}(next_state) when next_state is not on grid.
    """
    if x_query <= x_vals[0]:
        return y_vals[0]
    if x_query >= x_vals[-1]:
        return y_vals[-1]

    # Find interval containing x_query
    idx = np.searchsorted(x_vals, x_query) - 1
    x0, x1 = x_vals[idx], x_vals[idx + 1]
    y0, y1 = y_vals[idx], y_vals[idx + 1]

    # Linear interpolation formula
    return y0 + (x_query - x0) * (y1 - y0) / (x1 - x0)

def solve_dp() -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Main DP solver:
    - Discretizes state space X = 0,1,...,X_MAX
    - Computes F_k(X) and Y_k(X) for k=1..N_STEPS
    Returns:
        X_grid : array of state values
        F_tables : list where F_tables[k-1][i] = F_k(X_grid[i])
        Y_tables : list where Y_tables[k-1][i] = Y_k(X_grid[i])
    """
    # State grid
    X_grid = np.arange(0, X_MAX + GRID_STEP, GRID_STEP)

    # Storage for value functions and policies
    F_tables = []   # F_k arrays
    Y_tables = []   # Y_k arrays

    # Base case k = 1 
    F1 = np.zeros_like(X_grid)
    Y1 = np.zeros_like(X_grid)

    for i, X in enumerate(X_grid):
        if X == 0:
            F1[i] = 0.0
            Y1[i] = 0.0
            continue

        best_cost = float('inf')
        best_Y = 0.0
        # Brute‑force search over Y in [0, X] with step Y_STEP
        Y_candidates = np.arange(0.0, X + Y_STEP, Y_STEP)
        for Y in Y_candidates:
            cost = stage_cost(Y, X)
            if cost < best_cost:
                best_cost = cost
                best_Y = Y
        F1[i] = best_cost
        Y1[i] = best_Y

    F_tables.append(F1)
    Y_tables.append(Y1)

    # Recursive steps k = 2 .. N_STEPS 
    for k in range(2, N_STEPS + 1):
        Fk = np.zeros_like(X_grid)
        Yk = np.zeros_like(X_grid)
        F_prev = F_tables[-1]   # F_{k-1}

        for i, X in enumerate(X_grid):
            if X == 0:
                Fk[i] = 0.0
                Yk[i] = 0.0
                continue

            best_cost = float('inf')
            best_Y = 0.0
            Y_candidates = np.arange(0.0, X + Y_STEP, Y_STEP)

            for Y in Y_candidates:
                # Immediate cost of current stage
                immediate = stage_cost(Y, X)

                # Next state resource
                X_next = ALPHA * Y + BETA * (X - Y)

                # Future cost (interpolated from previous table)
                future = linear_interpolate(X_grid, F_prev, X_next)

                total = immediate + future
                if total < best_cost:
                    best_cost = total
                    best_Y = Y

            Fk[i] = best_cost
            Yk[i] = best_Y

        F_tables.append(Fk)
        Y_tables.append(Yk)

    return X_grid, F_tables, Y_tables

def recover_trajectory(X_initial: float, X_grid: np.ndarray,
                       Y_tables: List[np.ndarray]) -> List[float]:
    """
    Backward recursion to recover the optimal control sequence
    for a given initial resource X_initial.
    Returns list [y1, y2, ..., yN] in forward order.
    """
    trajectory = []
    X_current = X_initial

    # Process stages from N_STEPS down to 1
    for k in range(N_STEPS, 0, -1):
        Yk = Y_tables[k-1]     # policy for stage k
        # Interpolate optimal Y for current state X_current
        Y_opt = linear_interpolate(X_grid, Yk, X_current)
        trajectory.append(Y_opt)

        # Update resource for the next (previous) stage
        X_current = ALPHA * Y_opt + BETA * (X_current - Y_opt)

    return trajectory

# Main execution
if __name__ == "__main__":
    print("Numerical DP Solver for Nonlinear Problem\n")
    print(f"Parameters: alpha={ALPHA}, beta={BETA}, N={N_STEPS}, X_max={X_MAX}, grid step={GRID_STEP}\n")

    # Solve DP
    X_grid, F_tables, Y_tables = solve_dp()

    # Print table for X = 1 .. X_MAX
    print(f"{'X':>3} | {'F1':>8} {'Y1':>8} | {'F2':>8} {'Y2':>8} | {'F3':>8} {'Y3':>8} | {'F4':>8} {'Y4':>8}")
    print("-" * 75)
    for X_val in range(1, X_MAX + 1):
        idx = int(X_val / GRID_STEP)
        f1, y1 = F_tables[0][idx], Y_tables[0][idx]
        f2, y2 = F_tables[1][idx], Y_tables[1][idx]
        f3, y3 = F_tables[2][idx], Y_tables[2][idx]
        f4, y4 = F_tables[3][idx], Y_tables[3][idx]
        print(f"{X_val:3} | {f1:8.2f} {y1:8.2f} | {f2:8.2f} {y2:8.2f} | {f3:8.2f} {y3:8.2f} | {f4:8.2f} {y4:8.2f}")

    # Recover optimal trajectory for X = 14
    print(f"\nOptimal trajectory for initial resource X = {X_MAX}")
    traj = recover_trajectory(float(X_MAX), X_grid, Y_tables)
    for step, y_val in enumerate(traj, start=1):
        print(f"Step {step}: y_{step} = {y_val:.4f}")

    # Minimum total cost
    idx_start = int(X_MAX / GRID_STEP)
    min_cost = F_tables[-1][idx_start]
    print(f"\nMinimum total cost F_{N_STEPS}({X_MAX}) = {min_cost:.2f}")