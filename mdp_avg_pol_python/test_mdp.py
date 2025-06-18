"""
Test module for MDP Average Reward Policy Iteration
"""

import numpy as np
from pia import pia, pia_with_verbose
from main import create_example_mdp
from solver import solver, solver_gauss_jordan


def test_solver():
    """Test the linear system solver"""
    print("Testing linear system solver...")
    
    # Test case: 2x2 system
    G = np.array([
        [1, 0, 5],    # x = 5
        [0, 1, 3]     # y = 3
    ])
    
    x = solver(G)
    expected = np.array([5, 3])
    
    assert np.allclose(x, expected), f"Expected {expected}, got {x}"
    print("✓ Linear system solver test passed")


def test_policy_iteration():
    """Test the policy iteration algorithm"""
    print("Testing policy iteration algorithm...")
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Run policy iteration without verbose output
    policy, values, rho, iterations = pia_with_verbose(tpm, trm, verbose=False)
    
    # Check that we got reasonable results
    assert len(policy) == 2, f"Expected policy length 2, got {len(policy)}"
    assert len(values) == 2, f"Expected values length 2, got {len(values)}"
    assert iterations > 0, f"Expected positive iterations, got {iterations}"
    assert isinstance(rho, (int, float)), f"Expected numeric rho, got {type(rho)}"
    
    print(f"✓ Policy iteration test passed")
    print(f"  Optimal Policy: {policy}")
    print(f"  Optimal Values: {values}")
    print(f"  Average Reward: {rho}")
    print(f"  Iterations: {iterations}")


def test_mdp_consistency():
    """Test that the MDP matrices are consistent"""
    print("Testing MDP consistency...")
    
    tpm, trm = create_example_mdp()
    
    # Check shapes
    assert tpm.shape == (2, 2, 2), f"Expected tpm shape (2,2,2), got {tpm.shape}"
    assert trm.shape == (2, 2, 2), f"Expected trm shape (2,2,2), got {trm.shape}"
    
    # Check that transition probabilities sum to 1
    for action in range(2):
        for state in range(2):
            prob_sum = np.sum(tpm[action, state, :])
            assert np.isclose(prob_sum, 1.0), f"Probabilities don't sum to 1: {prob_sum}"
    
    print("✓ MDP consistency test passed")


def run_all_tests():
    """Run all tests"""
    print("Running MDP Average Reward Policy Iteration Tests")
    print("=" * 50)
    
    try:
        test_solver()
        test_mdp_consistency()
        test_policy_iteration()
        
        print("=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests() 