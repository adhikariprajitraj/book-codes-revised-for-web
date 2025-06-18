"""
Example script demonstrating how to use the MDP Average Reward Policy Iteration package
"""

import numpy as np
from pia import pia


def create_custom_mdp():
    """
    Create a custom MDP with 3 states and 2 actions
    """
    # Transition probability matrix (2 actions x 3 states x 3 states)
    tpm = np.array([
        # Action 0
        [
            [0.6, 0.3, 0.1],  # From state 0
            [0.2, 0.7, 0.1],  # From state 1
            [0.1, 0.2, 0.7]   # From state 2
        ],
        # Action 1
        [
            [0.8, 0.1, 0.1],  # From state 0
            [0.1, 0.8, 0.1],  # From state 1
            [0.1, 0.1, 0.8]   # From state 2
        ]
    ])
    
    # Transition reward matrix (2 actions x 3 states x 3 states)
    trm = np.array([
        # Action 0
        [
            [5, 2, -1],    # From state 0
            [3, 8, 1],     # From state 1
            [-2, 4, 10]    # From state 2
        ],
        # Action 1
        [
            [7, 1, 0],     # From state 0
            [2, 9, 3],     # From state 1
            [1, 3, 12]     # From state 2
        ]
    ])
    
    return tpm, trm


def print_mdp_analysis(tpm, trm, policy, values, rho, iterations):
    """
    Print a detailed analysis of the MDP solution
    """
    print("\n" + "="*60)
    print("MDP ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Number of states: {tpm.shape[1]}")
    print(f"Number of actions: {tpm.shape[0]}")
    print(f"Optimal average reward: {rho:.4f}")
    print(f"Convergence in {iterations} iterations")
    
    print("\nOptimal Policy:")
    for state in range(len(policy)):
        print(f"  State {state}: Action {policy[state]}")
    
    print("\nValue Function:")
    for state in range(len(values)):
        print(f"  State {state}: {values[state]:.4f}")
    
    print("\nPolicy Analysis:")
    for state in range(len(policy)):
        action = policy[state]
        print(f"\n  State {state} (Action {action}):")
        
        # Show transition probabilities for the chosen action
        print(f"    Transition probabilities: {tpm[action, state, :]}")
        
        # Show expected immediate reward
        expected_reward = np.sum(tpm[action, state, :] * trm[action, state, :])
        print(f"    Expected immediate reward: {expected_reward:.4f}")
        
        # Show expected next state value
        expected_next_value = np.sum(tpm[action, state, :] * values)
        print(f"    Expected next state value: {expected_next_value:.4f}")
        
        # Show total expected value (immediate + next state)
        total_expected = expected_reward + expected_next_value
        print(f"    Total expected value: {total_expected:.4f}")


def main():
    """
    Main function demonstrating the package usage
    """
    print("MDP Average Reward Policy Iteration - Custom Example")
    print("="*60)
    
    # Create a custom MDP
    print("Creating a custom 3-state, 2-action MDP...")
    tpm, trm = create_custom_mdp()
    
    # Verify MDP consistency
    print("Verifying MDP consistency...")
    for action in range(tpm.shape[0]):
        for state in range(tpm.shape[1]):
            prob_sum = np.sum(tpm[action, state, :])
            if not np.isclose(prob_sum, 1.0):
                print(f"Warning: Probabilities for action {action}, state {state} don't sum to 1: {prob_sum}")
    
    print("✓ MDP is consistent")
    
    # Run policy iteration
    print("\nRunning Policy Iteration Algorithm...")
    policy, values, rho, iterations = pia(tpm, trm)
    
    # Print detailed analysis
    print_mdp_analysis(tpm, trm, policy, values, rho, iterations)
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main() 