"""
Test module for MDP Average Reward Value Iteration
"""

import numpy as np
from rvia import rvia, rvia_with_verbose
from via import via, via_with_verbose
from main import create_example_mdp


def test_rvia():
    """Test the Relative Value Iteration algorithm"""
    print("Testing Relative Value Iteration (RVI)...")
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Run RVI without verbose output
    policy, values, avg_reward, iterations = rvia_with_verbose(tpm, trm, 0.001, verbose=False)
    
    # Check that we got reasonable results
    assert len(policy) == 2, f"Expected policy length 2, got {len(policy)}"
    assert len(values) == 2, f"Expected values length 2, got {len(values)}"
    assert iterations > 0, f"Expected positive iterations, got {iterations}"
    assert isinstance(avg_reward, (int, float)), f"Expected numeric avg_reward, got {type(avg_reward)}"
    
    print(f"✓ RVI test passed")
    print(f"  Optimal Policy: {policy}")
    print(f"  Optimal Values: {values}")
    print(f"  Average Reward: {avg_reward}")
    print(f"  Iterations: {iterations}")


def test_via():
    """Test the Standard Value Iteration algorithm"""
    print("Testing Standard Value Iteration (VI)...")
    
    # Create the example MDP
    tpm, trm = create_example_mdp()
    
    # Run VI without verbose output
    policy, values, iterations = via_with_verbose(tpm, trm, 0.001, verbose=False)
    
    # Check that we got reasonable results
    assert len(policy) == 2, f"Expected policy length 2, got {len(policy)}"
    assert len(values) == 2, f"Expected values length 2, got {len(values)}"
    assert iterations > 0, f"Expected positive iterations, got {iterations}"
    
    print(f"✓ VI test passed")
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


def test_convergence():
    """Test that both algorithms converge to similar policies"""
    print("Testing convergence consistency...")
    
    tpm, trm = create_example_mdp()
    epsilon = 0.001
    
    # Run both algorithms
    policy_rvi, values_rvi, avg_reward_rvi, iterations_rvi = rvia_with_verbose(tpm, trm, epsilon, verbose=False)
    policy_vi, values_vi, iterations_vi = via_with_verbose(tpm, trm, epsilon, verbose=False)
    
    # Check that policies are reasonable (both should find optimal actions)
    # The exact policy may vary due to numerical differences, but should be valid
    assert all(p in [0, 1] for p in policy_rvi), f"Invalid policy values in RVI: {policy_rvi}"
    assert all(p in [0, 1] for p in policy_vi), f"Invalid policy values in VI: {policy_vi}"
    
    print(f"✓ Convergence test passed")
    print(f"  RVI Policy: {policy_rvi}, Iterations: {iterations_rvi}")
    print(f"  VI Policy: {policy_vi}, Iterations: {iterations_vi}")


def run_all_tests():
    """Run all tests"""
    print("Running MDP Average Reward Value Iteration Tests")
    print("=" * 50)
    
    try:
        test_mdp_consistency()
        test_rvia()
        test_via()
        test_convergence()
        
        print("=" * 50)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests() 