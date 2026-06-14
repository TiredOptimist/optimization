"""
LP Problem for Microservices Production Planning
Variant 7: Maximize F = 13x1 + 15x2 + 10x3 + 12x4 + 14x5 + 18x6
subject to constraints.

Required libraries:
- scipy.optimize.linprog: main LP solver (method='highs')
- numpy: array operations
- pandas: formatted output
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Objective coefficients for maximization, negated because linprog minimizes
c = np.array([-13, -15, -10, -12, -14, -18])  # negative for linprog (minimization)

# Inequality constraints (<=)
A_ub = np.array([
    [3, 4, 2, 5, 2, 6],
    [2, 3, 1, 4, 1, 5],
    [15, 20, 10, 25, 8, 30],
    [0, 0, 0, 0, 1, 1]      # x5 + x6 <= 19
])
b_ub = np.array([250, 180, 1000, 19])

# Constraints of type ">=" need to be converted to "<=" by multiplying by -1
A_ub_ge = np.array([
    [-1, -1, -1, 0, 0, 0],
    [0, 0, 0, -1, -1, -1]
])
b_ub_ge = np.array([-20, -15])

# Equality constraints: None for this problem
A_eq = None
b_eq = None

# Append these to the existing inequality matrices
A_ub = np.vstack([A_ub, A_ub_ge])
b_ub = np.concatenate([b_ub, b_ub_ge])

# Variable bounds: x_j >= 0
bounds = [(0, None) for _ in range(6)]

# Solve primal LP
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method='highs')

# Check solution status
if result.success:
    primal_solution = result.x
    optimal_value = -result.fun   # revert sign for maximization
    print("Primal solution found")
else:
    print("No solution found:", result.message)
    exit()

# Extract dual variables (shadow prices)

# For scipy.optimize.linprog with method='highs', result.ineqlin.marginals
# gives dual variables for the minimization problem (min -F).
# To obtain true shadow prices for the original maximization problem:
#   - For original "≤" constraints: shadow price = -marginal
#   - For constraints originally "≥" (converted to "≤" by multiplying by -1):
#     shadow price = +marginal

shadow_prices = result.ineqlin.marginals  # length = number of inequality constraints

# First 4 constraints are original "≤" (DevOps, Budget, Compute quotas, Gateway limit)
# Their true shadow prices = -marginal.
# Last 2 constraints (indices 4,5) came from "≥" after multiplication by -1,
# so their shadow prices = +marginal (no change).
shadow_prices_corrected = shadow_prices.copy()
shadow_prices_corrected[:4] = -shadow_prices[:4]

# Now use shadow_prices_corrected for all further output and analysis
shadow_prices = shadow_prices_corrected  # replace with corrected version

# List constraints in order they appear in A_ub, b_ub:
constraint_names = [
    "DevOps-часы", 
    "Бюджет на облако", 
    "Вычислительные квоты", 
    "Лимит высоконагруженных шлюзов",
    "Мин. критические сервисы", 
    "Баланс инфраструктуры"
]


# Sensitivity analysis: change budget (second constraint) by +-15%
original_b = b_ub.copy()
perturbations = [-0.15, 0.15]
sensitivity_results = []

budget_index = 1  
for delta in perturbations:
    new_b = original_b.copy()
    new_b[budget_index] = original_b[budget_index] * (1 + delta)
    res = linprog(c, A_ub=A_ub, b_ub=new_b, bounds=bounds, method='highs')
    if res.success:
        new_obj = -res.fun
        change = new_obj - optimal_value
        rel_change = change / optimal_value
        sensitivity_results.append({
            "Constraint": constraint_names[budget_index],
            "Change %": f"{delta*100:+.0f}%",
            "New RHS": new_b[budget_index],
            "New Objective": new_obj,
            "Abs Change": change,
            "Rel Change %": f"{rel_change*100:+.2f}%"
        })

# Optimal plan
plan_df = pd.DataFrame({
    "Variable": [f"x{i+1}" for i in range(6)],
    "Optimal Value": primal_solution,
    "Coefficient in F": [-c[i] for i in range(6)]
})
print("\nOPTIMAL PLAN (Primal variables)")
print(plan_df)
print(f"\nMaximum objective value F_max = {optimal_value:.4f}")

# Shadow prices
dual_df = pd.DataFrame({
    "Constraint": constraint_names,
    "Shadow Price (Dual variable)": shadow_prices
})
print("\nDUAL VARIABLES (Shadow Prices / Marginal values)")
print(dual_df)

# Sensitivity
print("\nSENSITIVITY ANALYSIS (+-15% change in budgets)")
sens_df = pd.DataFrame(sensitivity_results)
print(sens_df.to_string(index=False))

