"""
Relative Value Iteration Algorithm for Discounted MDPs
Equivalent to rvi.c in the original C code

This implements relative value iteration where values are kept bounded by subtracting
a reference value (typically from state 0).
"""

import numpy as np
from constants import NS, NA, SMALL


def rvi(tpm, trm, d_factor, epsilon):
    """
    Relative Value Iteration for Discounted MDPs.
    The code assumes state 0 to be the subtraction factor state.
    
    Args:
        tpm: Transition probability matrix of shape (NA, NS, NS)
        trm: Transition reward matrix of shape (NA, NS, NS)
        d_factor: Discount factor (gamma)
        epsilon: Convergence threshold
        
    Returns:
        tuple: (optimal_policy, optimal_values, iterations)
    """
    # Determine termination factor
    epsilon = epsilon * (1 - d_factor) * 0.5 / d_factor
    
    # Initialize values
    value = np.zeros(NS)
    value_old = np.zeros(NS)
    policy = np.zeros(NS, dtype=int)
    
    done = True
    iter_count = 1
    
    while done:
        # Main loop of value iteration
        # Keeping a copy of the old value function
        value_old[:] = value[:]
        
        # Value Update
        for state in range(NS):
            # Value updated state by state
            best = SMALL
            
            for action in range(NA):
                # Find the best value for each state
                sum_val = 0
                for next_state in range(NS):
                    sum_val += (tpm[action, state, next_state] * 
                               (trm[action, state, next_state] + 
                                d_factor * value_old[next_state]))
                
                sum_val = sum_val - value_old[0]  # RVI subtraction
                
                if sum_val > best:
                    best = sum_val
                    policy[state] = action
                    value[state] = best
        
        # Determine the norm of the difference vector
        norm = -1
        for state in range(NS):
            if abs(value[state] - value_old[state]) > norm:
                norm = abs(value[state] - value_old[state])
        
        # Determine whether to terminate
        if norm < epsilon:
            # terminate
            done = False
        
        iter_count += 1
    
    # Display policy and value function
    for state in range(NS):
        print(f"Epsilon-optimal action for state {state}={policy[state]}")
    
    for state in range(NS):
        print(f"The value function for state {state}={value[state]}")
    
    print(f"The number of iterations needed to converge= {iter_count}")
    
    return policy, value, iter_count


def rvi_with_verbose(tpm, trm, d_factor, epsilon, verbose=True):
    """
    Relative Value Iteration with optional verbose output
    
    Args:
        tpm: Transition probability matrix
        trm: Transition reward matrix
        d_factor: Discount factor
        epsilon: Convergence threshold
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
        result = rvi(tpm, trm, d_factor, epsilon)
        return result
    finally:
        if not verbose:
            # Restore original print function
            builtins.print = original_print 