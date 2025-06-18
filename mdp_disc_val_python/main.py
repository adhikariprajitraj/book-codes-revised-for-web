"""
Main module for MDP Discounted Value Iteration
Equivalent to main.c in the original C code
"""

import numpy as np
from vi import vi
from gsvi import gsvi
from rvi import rvi


def create_example_mdp():
    """
    Create the example MDP from the original C code
    
    Returns:
        tuple: (tpm, trm) - transition probability and reward matrices
    """
    # Transition probability matrix (NA x NS x NS)
    # tpm[action][state][next_state] = P(next_state | state, action)
    tpm = np.array([
        # Action 0
        [
            [0.7, 0.3],  # From state 0: P(0|0,0)=0.7, P(1|0,0)=0.3
            [0.4, 0.6]   # From state 1: P(0|1,0)=0.4, P(1|1,0)=0.6
        ],
        # Action 1
        [
            [0.9, 0.1],  # From state 0: P(0|0,1)=0.9, P(1|0,1)=0.1
            [0.2, 0.8]   # From state 1: P(0|1,1)=0.2, P(1|1,1)=0.8
        ]
    ])
    
    # Transition reward matrix (NA x NS x NS)
    # trm[action][state][next_state] = R(state, action, next_state)
    trm = np.array([
        # Action 0
        [
            [6, -5],     # From state 0: R(0,0,0)=6, R(0,0,1)=-5
            [7, 12]      # From state 1: R(1,0,0)=7, R(1,0,1)=12
        ],
        # Action 1
        [
            [10, 17],    # From state 0: R(0,1,0)=10, R(0,1,1)=17
            [-14, 13]    # From state 1: R(1,1,0)=-14, R(1,1,1)=13
        ]
    ])
    
    return tpm, trm


def print_mdp_info(tpm, trm, d_factor, epsilon):
    """
    Print information about the MDP
    
    Args:
        tpm: Transition probability matrix
        trm: Transition reward matrix
        d_factor: Discount factor
        epsilon: Convergence threshold
    """
    print("=" * 50)
    print("MDP INFORMATION")
    print("=" * 50)
    print(f"Number of states: {tpm.shape[1]}")
    print(f"Number of actions: {tpm.shape[0]}")
    print(f"Discount factor: {d_factor}")
    print(f"Convergence threshold: {epsilon}")
    print()
    
    print("Transition Probabilities:")
    for action in range(tpm.shape[0]):
        print(f"  Action {action}:")
        for state in range(tpm.shape[1]):
            print(f"    From state {state}: {tpm[action, state, :]}")
    print()
    
    print("Transition Rewards:")
    for action in range(trm.shape[0]):
        print(f"  Action {action}:")
        for state in range(trm.shape[1]):
            print(f"    From state {state}: {trm[action, state, :]}")
    print("=" * 50)
    print()


def main():
    """
    Main function that sets up the MDP and runs discounted value iteration
    """
    print("MDP Discounted Value Iteration")
    print("Python Implementation")
    print()
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Set parameters
    d_factor = 0.8
    epsilon = 0.001
    
    # Print MDP information
    print_mdp_info(tpm, trm, d_factor, epsilon)
    
    # Run Gauss-Seidel Value Iteration (as in original main.c)
    print("Running Gauss-Seidel Value Iteration (GSVI) Algorithm...")
    print(f"Discount factor (gamma): {d_factor}")
    print(f"Convergence threshold (epsilon): {epsilon}")
    print()
    
    optimal_policy, optimal_values, iterations = gsvi(tpm, trm, d_factor, epsilon)
    
    print()
    print("=" * 50)
    print("FINAL RESULTS - GSVI")
    print("=" * 50)
    print(f"Optimal Policy: {optimal_policy}")
    print(f"Optimal Values: {optimal_values}")
    print(f"Total Iterations: {iterations}")
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("COMPARISON WITH OTHER VALUE ITERATION ALGORITHMS")
    print("=" * 50)
    
    # Run Standard Value Iteration for comparison
    print("Running Standard Value Iteration (VI) Algorithm...")
    print()
    
    vi_policy, vi_values, vi_iterations = vi(tpm, trm, d_factor, epsilon)
    
    print()
    print("=" * 50)
    print("FINAL RESULTS - VI")
    print("=" * 50)
    print(f"Optimal Policy: {vi_policy}")
    print(f"Optimal Values: {vi_values}")
    print(f"Total Iterations: {vi_iterations}")
    print("=" * 50)
    
    # Run Relative Value Iteration for comparison
    print("\nRunning Relative Value Iteration (RVI) Algorithm...")
    print()
    
    rvi_policy, rvi_values, rvi_iterations = rvi(tpm, trm, d_factor, epsilon)
    
    print()
    print("=" * 50)
    print("FINAL RESULTS - RVI")
    print("=" * 50)
    print(f"Optimal Policy: {rvi_policy}")
    print(f"Optimal Values: {rvi_values}")
    print(f"Total Iterations: {rvi_iterations}")
    print("=" * 50)
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Policies match (VI=GSVI): {np.array_equal(vi_policy, optimal_policy)}")
    print(f"Policies match (VI=RVI): {np.array_equal(vi_policy, rvi_policy)}")
    print(f"VI iterations: {vi_iterations}")
    print(f"GSVI iterations: {iterations}")
    print(f"RVI iterations: {rvi_iterations}")
    print("=" * 50)


if __name__ == "__main__":
    main() 