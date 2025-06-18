"""
Main module for MDP Discounted Policy Iteration
Equivalent to main.c in the original C code
"""

import numpy as np
from pid import pid


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


def print_mdp_info(tpm, trm, d_factor):
    """
    Print information about the MDP
    
    Args:
        tpm: Transition probability matrix
        trm: Transition reward matrix
        d_factor: Discount factor
    """
    print("=" * 50)
    print("MDP INFORMATION")
    print("=" * 50)
    print(f"Number of states: {tpm.shape[1]}")
    print(f"Number of actions: {tpm.shape[0]}")
    print(f"Discount factor: {d_factor}")
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
    Main function that sets up the MDP and runs discounted policy iteration
    """
    print("MDP Discounted Policy Iteration")
    print("Python Implementation")
    print()
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Set discount factor
    d_factor = 0.8
    
    # Print MDP information
    print_mdp_info(tpm, trm, d_factor)
    
    # Run discounted policy iteration
    print("Running Discounted Policy Iteration Algorithm...")
    print(f"Discount factor (gamma): {d_factor}")
    print()
    
    optimal_policy, optimal_values, iterations = pid(tpm, trm, d_factor)
    
    print()
    print("=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Optimal Policy: {optimal_policy}")
    print(f"Optimal Values: {optimal_values}")
    print(f"Total Iterations: {iterations}")
    print("=" * 50)


if __name__ == "__main__":
    main() 