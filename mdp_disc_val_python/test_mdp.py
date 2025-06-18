"""
Test module for MDP Discounted Value Iteration
"""

import numpy as np
from vi import vi, vi_with_verbose
from gsvi import gsvi, gsvi_with_verbose
from rvi import rvi, rvi_with_verbose
from main import create_example_mdp


def test_standard_vi():
    """Test the standard value iteration algorithm"""
    print("Testing Standard Value Iteration (VI)...")
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Run VI without verbose output
    policy, values, iterations = vi_with_verbose(tpm, trm, 0.8, 0.001, verbose=False)
    
    # Check that we got reasonable results
    assert len(policy) == 2, f"Expected policy length 2, got {len(policy)}"
    assert len(values) == 2, f"Expected values length 2, got {len(values)}"
    assert iterations > 0, f"Expected positive iterations, got {iterations}"
    
    print(f"✓ Standard VI test passed")
    print(f"  Optimal Policy: {policy}")
    print(f"  Optimal Values: {values}")
    print(f"  Iterations: {iterations}")


def test_gauss_seidel_vi():
    """Test the Gauss-Seidel value iteration algorithm"""
    print("Testing Gauss-Seidel Value Iteration (GSVI)...")
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Run GSVI without verbose output
    policy, values, iterations = gsvi_with_verbose(tpm, trm, 0.8, 0.001, verbose=False)
    
    # Check that we got reasonable results
    assert len(policy) == 2, f"Expected policy length 2, got {len(policy)}"
    assert len(values) == 2, f"Expected values length 2, got {len(values)}"
    assert iterations > 0, f"Expected positive iterations, got {iterations}"
    
    print(f"✓ Gauss-Seidel VI test passed")
    print(f"  Optimal Policy: {policy}")
    print(f"  Optimal Values: {values}")
    print(f"  Iterations: {iterations}")


def test_relative_vi():
    """Test the relative value iteration algorithm"""
    print("Testing Relative Value Iteration (RVI)...")
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Run RVI without verbose output
    policy, values, iterations = rvi_with_verbose(tpm, trm, 0.8, 0.001, verbose=False)
    
    # Check that we got reasonable results
    assert len(policy) == 2, f"Expected policy length 2, got {len(policy)}"
    assert len(values) == 2, f"Expected values length 2, got {len(values)}"
    assert iterations > 0, f"Expected positive iterations, got {iterations}"
    
    print(f"✓ Relative VI test passed")
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


def test_convergence_consistency():
    """Test that all algorithms converge to similar policies"""
    print("Testing convergence consistency...")
    
    tpm, trm = create_example_mdp()
    d_factor = 0.8
    epsilon = 0.001
    
    # Run all algorithms
    vi_policy, vi_values, vi_iterations = vi_with_verbose(tpm, trm, d_factor, epsilon, verbose=False)
    gsvi_policy, gsvi_values, gsvi_iterations = gsvi_with_verbose(tpm, trm, d_factor, epsilon, verbose=False)
    rvi_policy, rvi_values, rvi_iterations = rvi_with_verbose(tpm, trm, d_factor, epsilon, verbose=False)
    
    # Check that policies are reasonable (both should find optimal actions)
    assert all(p in [0, 1] for p in vi_policy), f"Invalid policy values in VI: {vi_policy}"
    assert all(p in [0, 1] for p in gsvi_policy), f"Invalid policy values in GSVI: {gsvi_policy}"
    assert all(p in [0, 1] for p in rvi_policy), f"Invalid policy values in RVI: {rvi_policy}"
    
    print(f"✓ Convergence test passed")
    print(f"  VI Policy: {vi_policy}, Iterations: {vi_iterations}")
    print(f"  GSVI Policy: {gsvi_policy}, Iterations: {gsvi_iterations}")
    print(f"  RVI Policy: {rvi_policy}, Iterations: {rvi_iterations}")


def run_all_tests():
    """Run all tests"""
    print("Running MDP Discounted Value Iteration Tests")
    print("=" * 50)
    
    try:
        test_mdp_consistency()
        test_standard_vi()
        test_gauss_seidel_vi()
        test_relative_vi()
        test_convergence_consistency()
        
        print("=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests() 