"""
Numerical Dynamic Programming for Equipment Replacement Problem
Variant: Maximize total profit over 6 stages, initial age 3,
         profit function R(t) = 22 - 1.5t,
         replacement cost C(t) = 7 + 1.8t.
Context: Replacement of data storage servers.

Required libraries:
- numpy: for array storage and operations
- typing: type hints (List, Tuple)
"""

import numpy as np
from typing import List, Tuple


def compute_F(N: int, Tmax: int, R, C, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DP tables F[k][t] and decision[k][t] for k=1..N, t=0..Tmax.

    Parameters:
        N: planning horizon (number of years)
        Tmax: maximum age considered (ages > Tmax yield zero profit)
        R: profit function R(t)
        C: replacement cost function C(t)
        verbose: if True, print detailed step-by-step calculations

    Returns:
        F: 2D numpy array of shape (N+1) x (Tmax+1), F[k][t] = max profit
           with k years remaining and current age t.
        decision: 2D numpy array of same shape, decision[k][t] = 0 if Keep,
                  1 if Replace. (decision[0] is unused)
    """
    F = np.zeros((N + 1, Tmax + 1), dtype=float)
    decision = np.zeros((N + 1, Tmax + 1), dtype=int)

    if verbose:
        print("\n" + "=" * 80)
        print("FORWARD PASS: Filling Bellman tables")
        print("=" * 80)

    # Base case: k = 1 (last year)
    if verbose:
        print(f"\nStep k = 1 (1 year remaining):")
        print(f"{'t':<5} {'Keep':<12} {'Replace':<12} {'F_1(t)':<10} {'Decision':<10}")
        print("-" * 55)

    for t in range(Tmax + 1):
        keep_val = R(t)
        replace_val = R(0) - C(t)
        if keep_val >= replace_val:
            F[1][t] = keep_val
            decision[1][t] = 0   # Keep
            act = 'Keep'
        else:
            F[1][t] = replace_val
            decision[1][t] = 1   # Replace
            act = 'Replace'

        if verbose:
            print(f"{t:<5} {keep_val:<12.2f} {replace_val:<12.2f} {F[1][t]:<10.2f} {act:<10}")

    # Recursive steps: k = 2 .. N
    for k in range(2, N + 1):
        if verbose:
            print(f"\nStep k = {k} ({k} years remaining):")
            print(f"{'t':<5} {'Keep (R(t)+F_{k-1}(t+1))':<30} {'Replace (R(0)-C(t)+F_{k-1}(1))':<35} {'F_{k}(t)':<12} {'Decision':<10}")
            print("-" * 90)

        for t in range(Tmax + 1):
            # Option 1: Keep the machine
            if t + 1 <= Tmax:
                keep_val = R(t) + F[k - 1][t + 1]
                keep_str = f"R({t}) + F_{k-1}({t+1}) = {R(t):.2f} + {F[k-1][t+1]:.2f}"
            else:
                keep_val = R(t)  # F[k-1][t+1] = 0 for t+1 > Tmax
                keep_str = f"R({t}) + 0 = {R(t):.2f}"

            # Option 2: Replace the machine
            replace_val = R(0) - C(t) + F[k - 1][1]
            replace_str = f"R(0)-C({t}) + F_{k-1}(1) = {R(0):.2f}-{C(t):.2f} + {F[k-1][1]:.2f}"

            if keep_val >= replace_val:
                F[k][t] = keep_val
                decision[k][t] = 0
                act = 'Keep'
            else:
                F[k][t] = replace_val
                decision[k][t] = 1
                act = 'Replace'

            if verbose:
                print(f"{t:<5} {keep_str:<30} {replace_str:<35} {F[k][t]:<12.2f} {act:<10}")

    return F, decision


def recover_policy(decision: np.ndarray, t0: int, N: int, Tmax: int, verbose: bool = False) -> List[str]:
    """
    Recover the optimal sequence of actions (Keep / Replace) for the given
    initial age t0 and horizon N.

    Parameters:
        decision: decision table from compute_F
        t0: initial age at the beginning of the first year
        N: planning horizon
        Tmax: maximum age considered (used to handle out-of-range ages)
        verbose: if True, print detailed backtracking steps

    Returns:
        List of strings, length N, each element is 'Keep' or 'Replace'.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("BACKWARD PASS: Recovering optimal policy")
        print("=" * 80)

    actions = []
    t = t0
    for k in range(N, 0, -1):
        # If age exceeds Tmax, the machine is worthless; we must replace.
        if t > Tmax:
            act = 1
        else:
            act = decision[k][t]

        if verbose:
            action_str = 'Keep' if act == 0 else 'Replace'
            year = N - k + 1
            print(f"Year {year} (k={k}, age={t}): decision = {action_str}")

        if act == 0:
            actions.append('Keep')
            t = t + 1
        else:
            actions.append('Replace')
            t = 1   # after replacement, next year's age is 1

    return actions


def print_detailed_tables(F: np.ndarray, decision: np.ndarray, N: int, Tmax: int) -> None:
    """Print F and decision tables in a readable matrix format."""
    print("\n" + "=" * 80)
    print("DETAILED DP TABLES")
    print("=" * 80)

    # Print F table
    print("\nF[k][t] table (max profit with k years remaining, age t):")
    header = "k\\t"
    print("{:<6}".format(header), end="")
    for t in range(Tmax + 1):
        print(f"{t:>8}", end="")
    print()

    for k in range(1, N + 1):
        print(f"k={k:<4}", end="")
        for t in range(Tmax + 1):
            print(f"{F[k][t]:>8.2f}", end="")
        print()

    # Print Decision table
    print("\nDecision table (0=Keep, 1=Replace):")
    print("{:<6}".format(header), end="")
    for t in range(Tmax + 1):
        print(f"{t:>8}", end="")
    print()

    for k in range(1, N + 1):
        print(f"k={k:<4}", end="")
        for t in range(Tmax + 1):
            print(f"{int(decision[k][t]):>8}", end="")
        print()


def simulate_policy(policy: List[str], R, C, t0: int, N: int, verbose: bool = False) -> float:
    """
    Simulate the policy to compute actual total profit.

    Parameters:
        policy: list of actions ('Keep' or 'Replace') in chronological order
        R: profit function
        C: replacement cost function
        t0: initial age
        N: planning horizon
        verbose: if True, print detailed simulation steps

    Returns:
        total: total profit from following the policy
    """
    if verbose:
        print("\n" + "=" * 80)
        print("POLICY SIMULATION")
        print("=" * 80)
        print(f"{'Year':<6} {'Action':<10} {'Age':<6} {'Calculation':<25} {'Profit':<10}")
        print("-" * 65)

    t = t0
    total = 0.0

    for year in range(1, N + 1):
        act = policy[year - 1]

        if act == 'Keep':
            profit = R(t)
            total += profit
            if verbose:
                print(f"{year:<6} {'Keep':<10} {t:<6} R({t}) = {profit:.2f} {'':<12} {profit:<10.2f}")
            t = t + 1
        else:
            profit = R(0) - C(t)
            total += profit
            if verbose:
                print(f"{year:<6} {'Replace':<10} {t:<6} R(0)-C({t}) = {R(0):.2f}-{C(t):.2f} {'':<12} {profit:<10.2f}")
            t = 1

    if verbose:
        print("-" * 65)
        print(f"{'Total':<6} {'':<10} {'':<6} {'':<25} {total:<10.2f}")

    return total


def main():
    # Define profit and cost functions for this variant
    def R(t: float) -> float:
        return 22.0 - 1.5 * t

    def C(t: float) -> float:
        return 7.0 + 1.8 * t

    N = 6
    Tmax = 7
    t0 = 3

    print("=" * 80)
    print("DYNAMIC PROGRAMMING FOR EQUIPMENT REPLACEMENT")
    print("Context: Storage Server Replacement")
    print("=" * 80)
    print(f"Horizon N = {N} years")
    print(f"Maximum age considered Tmax = {Tmax}")
    print(f"Initial age t0 = {t0}")
    print(f"Profit function R(t) = 22 - 1.5t")
    print(f"Replacement cost C(t) = 7 + 1.8t")
    print("=" * 80)

    # Compute DP tables with detailed output
    F, decision = compute_F(N, Tmax, R, C, verbose=True)

    # Print detailed tables in matrix form
    print_detailed_tables(F, decision, N, Tmax)

    # Recover optimal policy with detailed backtracking
    policy = recover_policy(decision, t0, N, Tmax, verbose=True)

    # Print policy summary
    print("\n" + "=" * 80)
    print("OPTIMAL POLICY SUMMARY")
    print("=" * 80)
    print(f"Initial age: {t0}")
    print(f"Policy by year: {' -> '.join(policy)}")
    print(f"Maximum total profit: F[{N}][{t0}] = {F[N][t0]:.2f}")

    # Simulate policy with detailed output
    simulated_total = simulate_policy(policy, R, C, t0, N, verbose=True)

    # Verify consistency
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"DP maximum profit F[{N}][{t0}] = {F[N][t0]:.2f}")
    print(f"Simulated total profit = {simulated_total:.2f}")
    print(f"Match: {'YES' if abs(F[N][t0] - simulated_total) < 1e-9 else 'NO'}")


if __name__ == "__main__":
    main()