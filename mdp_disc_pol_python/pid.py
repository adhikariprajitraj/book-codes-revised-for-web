"""
Policy Iteration Algorithm for Discounted MDPs
Equivalent to pid.c in the original C code
"""

import numpy as np
from constants import NS, NA, SMALL
from solver import solver


def pid(tpm, trm, d_factor):
    """
    Policy Iteration for Discounted Reward MDPs.
    
    Args:
        tpm: Transition probability matrix of shape (NA, NS, NS)
        trm: Transition reward matrix of shape (NA, NS, NS)
        d_factor: Discount factor (gamma)
        
    Returns:
        tuple: (optimal_policy, optimal_values, iterations)
    """
    # Start with an arbitrary policy
    policy = np.zeros(NS, dtype=int)
    
    iteration = 0
    done = True
    
    while done:
        # As long as two consecutive policies do not become identical
        
        # 1. Policy evaluation stage
        # The linear equations to be solved are Gx=0.
        # Initializing a part of the G Matrix.
        G = np.zeros((NS, NS + 1))
        
        for row in range(NS):
            for col in range(NS):
                if row == col:
                    G[row, col] = 1 - d_factor * tpm[policy[row], row, col]
                else:
                    G[row, col] = -(d_factor * tpm[policy[row], row, col])
        
        # Initializing the (NS+1)th column of G matrix
        for state in range(NS):
            sum_val = 0.0
            for next_state in range(NS):
                sum_val += (tpm[policy[state], state, next_state] * 
                           trm[policy[state], state, next_state])
            G[state, NS] = sum_val
        
        # Solve the linear system
        x = solver(G)
        
        for state in range(NS):
            print(f"Value for state {state} in current iteration is {x[state]}")
        
        # 2. Policy improvement stage
        done = False
        
        for state in range(NS):
            large = SMALL
            best_action = 0
            
            for action in range(NA):
                # Determine the best action for the state
                sum_val = 0
                
                for next_state in range(NS):
                    sum_val += (tpm[action, state, next_state] * 
                               (trm[action, state, next_state] + 
                                d_factor * x[next_state]))
                
                if sum_val > large:
                    large = sum_val
                    best_action = action
            
            if policy[state] != best_action:
                # Policy has improved; record new action
                policy[state] = best_action
                done = True  # To ensure that one more iteration is done
        
        iteration += 1
    
    print(f"Number of iterations needed: {iteration}")
    
    for state in range(NS):
        print(f"Optimal action for state {state} is {policy[state]}")
    
    for state in range(NS):
        print(f"Optimal value for state {state} is {x[state]}")
    
    return policy, x, iteration


def pid_with_verbose(tpm, trm, d_factor, verbose=True):
    """
    Policy Iteration with optional verbose output
    
    Args:
        tpm: Transition probability matrix
        trm: Transition reward matrix
        d_factor: Discount factor
        verbose: Whether to print detailed output
        
    Returns:
        tuple: (optimal_policy, optimal_values, iterations)
    """
    if not verbose:
        # Redirect print to a dummy function
        import builtins
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
    
    try:
        result = pid(tpm, trm, d_factor)
        return result
    finally:
        if not verbose:
            # Restore original print function
            builtins.print = original_print 