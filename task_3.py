"""
Johnson's algorithm for 8 jobs on two machines (A: training, B: validation).
Given A_i and B_i, find optimal order minimizing makespan, compute B idle time,
and compare with original order (1..8).
"""

def johnson_order(A, B):
    """
    Johnson's algorithm for two machines.
    Returns the optimal job order (list of indices, 0-based).
    """
    n = len(A)
    tasks = list(range(n))          # list of unscheduled jobs
    order = [None] * n              # final order (filled from both ends)
    left = 0                        # next position from the start
    right = n - 1                   # next position from the end

    while tasks:
        # Find the minimum processing time among remaining A[i] and B[i]
        min_time = float('inf')
        min_task = None
        min_machine = None          # 'A' or 'B'

        for i in tasks:
            if A[i] < min_time:
                min_time = A[i]
                min_task = i
                min_machine = 'A'
            if B[i] < min_time:
                min_time = B[i]
                min_task = i
                min_machine = 'B'

        # Place the job according to Johnson's rule
        if min_machine == 'A':
            order[left] = min_task
            left += 1
        else:  # min on machine B
            order[right] = min_task
            right -= 1

        tasks.remove(min_task)

    return order


def compute_makespan_and_idle(order, A, B):
    """
    For a given job order (list of indices), computes:
        - makespan (total completion time)
        - idle time of machine B (total waiting time)
    Returns (makespan, idle_time).
    """
    time_A = 0      # time when machine A finishes its last job
    time_B = 0      # time when machine B finishes its last job
    idle_total = 0

    for i in order:
        end_A = time_A + A[i]               # when job i is ready on A
        start_B = max(end_A, time_B)        # B can start only after end_A and when free
        idle_total += start_B - time_B      # idle time of B before this job
        time_A = end_A
        time_B = start_B + B[i]             # completion time on B

    return time_B, idle_total


def print_schedule(order, A, B, title):
    """Prints the schedule in a table format."""
    print(f"\n{title}")
    print("Job\tA_i\tB_i\tStartA\tEndA\tStartB\tEndB\tB_idle")
    print("-" * 70)

    time_A = 0
    time_B = 0
    for idx, task in enumerate(order):
        end_A = time_A + A[task]
        start_B = max(end_A, time_B)
        idle = start_B - time_B
        end_B = start_B + B[task]

        print(f"{task+1}\t{A[task]}\t{B[task]}\t{time_A}\t{end_A}\t{start_B}\t{end_B}\t{idle}")
        time_A = end_A
        time_B = end_B

    print(f"\nMakespan = {time_B} min")
    print(f"Total idle time of machine B = {time_B - sum(B)} min")

def main():
    # Input data (training → validation)
    A = [5, 3, 8, 2, 7, 4, 6, 9]   # training time (machine A)
    B = [4, 7, 3, 8, 1, 6, 2, 5]   # validation time (machine B)

    # 1. Original order (1,2,3,4,5,6,7,8) -> indices 0..7
    original_order = list(range(8))
    makespan_orig, idle_orig = compute_makespan_and_idle(original_order, A, B)

    # 2. Optimal order by Johnson's algorithm
    optimal_order = johnson_order(A, B)
    makespan_opt, idle_opt = compute_makespan_and_idle(optimal_order, A, B)

    # Print results
    print("JOHNSON'S PROBLEM (model training -> dataset validation)")

    print_schedule(original_order, A, B, "ORIGINAL ORDER (1,2,3,4,5,6,7,8)")
    print_schedule(optimal_order, A, B, "OPTIMAL ORDER (Johnson's algorithm)")

    print("\nCOMPARISON OF RESULTS")
    print(f"{'Metric':<30} {'Original':<12} {'Optimal':<12} {'Change'}")
    print("-" * 70)
    print(f"{'Makespan (min)':<30} {makespan_orig:<12} {makespan_opt:<12} {makespan_opt - makespan_orig:+d}")
    print(f"{'B idle time (min)':<30} {idle_orig:<12} {idle_opt:<12} {idle_opt - idle_orig:+d}")
    print(f"{'Sum of B (min)':<30} {sum(B):<12} {sum(B):<12} {'-'}")

if __name__ == "__main__":
    main()