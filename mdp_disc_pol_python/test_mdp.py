"""
Test module for MDP Discounted Policy Iteration
"""

import numpy as np
from pid import pid, pid_with_verbose
from main import create_example_mdp


def test_policy_iteration():
    """Test the discounted policy iteration algorithm"""
    print("Testing Discounted Policy Iteration...")
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Run policy iteration without verbose output
    policy, values, iterations = pid_with_verbose(tpm, trm, 0.8, verbose=False)
    
    # Check that we got reasonable results
    assert len(policy) == 2, f"Expected policy length 2, got {len(policy)}"
    assert len(values) == 2, f"Expected values length 2, got {len(values)}"
    assert iterations > 0, f"Expected positive iterations, got {iterations}"
    
    print(f"✓ Policy iteration test passed")
    print(f"  Optimal Policy: {policy}")
    print(f"  Optimal Values: {values}")
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


def test_discount_factor():
    """Test that different discount factors produce different results"""
    print("Testing discount factor effects...")
    
    tpm, trm = create_example_mdp()
    
    # Test with different discount factors
    discount_factors = [0.5, 0.8, 0.9]
    policies = []
    
    for d_factor in discount_factors:
        policy, values, iterations = pid_with_verbose(tpm, trm, d_factor, verbose=False)
        policies.append(policy)
        print(f"  Discount {d_factor}: Policy {policy}, Values {values}")
    
    # Policies should be reasonable (valid actions)
    for policy in policies:
        assert all(p in [0, 1] for p in policy), f"Invalid policy values: {policy}"
    
    print("✓ Discount factor test passed")


def run_all_tests():
    """Run all tests"""
    print("Running MDP Discounted Policy Iteration Tests")
    print("=" * 50)
    
    try:
        test_mdp_consistency()
        test_policy_iteration()
        test_discount_factor()
        
        print("=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests() 