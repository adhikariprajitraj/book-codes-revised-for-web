"""
Example script demonstrating how to use the MDP Average Reward Value Iteration package
"""

import numpy as np
from rvia import rvia
from via import via


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


def compare_algorithms(tpm, trm, epsilon):
    """
    Compare RVI and VI algorithms with the same MDP and epsilon
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH EPSILON = {epsilon}")
    print(f"{'='*60}")
    
    # Run RVI
    print("\nRunning Relative Value Iteration (RVI)...")
    policy_rvi, values_rvi, avg_reward_rvi, iterations_rvi = rvia(tpm, trm, epsilon)
    
    print(f"\nRVI Results:")
    print(f"  Optimal Policy: {policy_rvi}")
    print(f"  Optimal Values: {values_rvi}")
    print(f"  Average Reward: {avg_reward_rvi:.4f}")
    print(f"  Iterations: {iterations_rvi}")
    
    # Run VI
    print("\nRunning Standard Value Iteration (VI)...")
    policy_vi, values_vi, iterations_vi = via(tpm, trm, epsilon)
    
    print(f"\nVI Results:")
    print(f"  Optimal Policy: {policy_vi}")
    print(f"  Optimal Values: {values_vi}")
    print(f"  Iterations: {iterations_vi}")
    
    # Compare
    print(f"\nComparison:")
    print(f"  Policies match: {np.array_equal(policy_rvi, policy_vi)}")
    print(f"  RVI iterations: {iterations_rvi}")
    print(f"  VI iterations: {iterations_vi}")
    print(f"  Iteration difference: {abs(iterations_rvi - iterations_vi)}")


def analyze_convergence():
    """
    Analyze how different epsilon values affect convergence
    """
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    tpm, trm = create_custom_mdp()
    
    epsilon_values = [0.1, 0.01, 0.001, 0.0001]
    
    print("\nEpsilon | RVI Iterations | VI Iterations | Policy Match")
    print("-" * 55)
    
    for epsilon in epsilon_values:
        # Run both algorithms
        policy_rvi, values_rvi, avg_reward_rvi, iterations_rvi = rvia(tpm, trm, epsilon)
        policy_vi, values_vi, iterations_vi = via(tpm, trm, epsilon)
        
        policies_match = np.array_equal(policy_rvi, policy_vi)
        
        print(f"{epsilon:7.4f} | {iterations_rvi:13d} | {iterations_vi:12d} | {policies_match}")


def main():
    """
    Main function demonstrating the package usage
    """
    print("MDP Average Reward Value Iteration - Custom Example")
    print("=" * 60)
    
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
    
    # Compare algorithms with different epsilon values
    epsilon_values = [0.01, 0.001]
    for epsilon in epsilon_values:
        compare_algorithms(tpm, trm, epsilon)
    
    # Analyze convergence
    analyze_convergence()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 