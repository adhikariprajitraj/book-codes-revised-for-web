"""
Policy Iteration Algorithm for Average Reward MDPs
Equivalent to pia.c in the original C code
"""

import numpy as np
from constants import NS, NA, SMALL
from solver import solver


def pia(tpm, trm):
    """
    Policy Iteration for Average Reward MDPs.
    The strategy adopted for the under-determined linear system is
    to replace first value by 0.
    
    Args:
        tpm: Transition probability matrix of shape (NA, NS, NS)
        trm: Transition reward matrix of shape (NA, NS, NS)
        
    Returns:
        tuple: (optimal_policy, optimal_values, average_reward, iterations)
    """
    # Start with an arbitrary policy
    policy = np.zeros(NS, dtype=int)
    
    iteration = 0
    done = True
    
    while done:
        # 1. Policy evaluation stage
        # The linear equations to be solved are Gx=0.
        # Initializing a part of the G Matrix.
        G = np.zeros((NS, NS + 1))
        
        for row in range(NS):
            for col in range(NS):
                if col == 0:
                    # Because the first value is replaced by rho
                    G[row, col] = 1
                else:
                    if row == col:
                        G[row, col] = 1 - tpm[policy[row], row, col]
                    else:
                        G[row, col] = -tpm[policy[row], row, col]
        
        # Initializing the (NS+1)th column of G matrix
        for state in range(NS):
            sum_val = 0.0
            for next_state in range(NS):
                sum_val += (tpm[policy[state], state, next_state] * 
                           trm[policy[state], state, next_state])
            G[state, NS] = sum_val
        
        # Solve the linear system
        x = solver(G)
        
        # Determine the average reward
        rho = x[0]
        
        print(f"Average reward in iteration {iteration}: {rho}")
        
        # The first value is set to 0
        x[0] = 0
        
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
                               (trm[action, state, next_state] + x[next_state]))
                
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
    
    return policy, x, rho, iteration


def pia_with_verbose(tpm, trm, verbose=True):
    """
    Policy Iteration with optional verbose output
    
    Args:
        tpm: Transition probability matrix
        trm: Transition reward matrix
        verbose: Whether to print detailed output
        
    Returns:
        tuple: (optimal_policy, optimal_values, average_reward, iterations)
    """
    if not verbose:
        # Redirect print to a dummy function
        import builtins
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
    
    try:
        result = pia(tpm, trm)
        return result
    finally:
        if not verbose:
            # Restore original print function
            builtins.print = original_print 