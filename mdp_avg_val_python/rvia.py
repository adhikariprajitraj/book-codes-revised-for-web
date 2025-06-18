"""
Relative Value Iteration Algorithm for Average Reward MDPs
Equivalent to rvia.c in the original C code

This implements relative value iteration where values are kept bounded by subtracting
a reference value (typically from state 0).
"""

import numpy as np
from constants import NS, NA, SMALL


def rvia(tpm, trm, epsilon):
    """
    Relative Value Iteration for Average Reward MDPs.
    The code assumes state 0 to be the subtraction factor state.
    
    Args:
        tpm: Transition probability matrix of shape (NA, NS, NS)
        trm: Transition reward matrix of shape (NA, NS, NS)
        epsilon: Convergence threshold
        
    Returns:
        tuple: (optimal_policy, optimal_values, average_reward, iterations)
    """
    # Initialize values
    value = np.zeros(NS)
    value_old = np.zeros(NS)
    policy = np.zeros(NS, dtype=int)
    
    done = True
    iter_count = 0
    
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
                               (trm[action, state, next_state] + value_old[next_state]))
                
                if sum_val > best:
                    best = sum_val
                    policy[state] = action
                    value[state] = best
        
        sub_factor = value[0]
        
        # RVI subtraction - subtract the reference value to keep values bounded
        for state in range(NS):
            value[state] = value[state] - sub_factor
        
        # Determine the span of the difference vector
        max_val = value[0] - value_old[0]
        min_val = value[0] - value_old[0]
        
        for state in range(1, NS):
            if value[state] - value_old[state] > max_val:
                max_val = value[state] - value_old[state]
            if value[state] - value_old[state] < min_val:
                min_val = value[state] - value_old[state]
        
        span = max_val - min_val
        print(f"span of difference vector={span}")
        print(f"Average reward estimate= {sub_factor}")
        
        # Determine whether to terminate
        if span < epsilon:
            # terminate
            done = False
        
        for state in range(NS):
            print(f"The value function for state {state}={value[state]}")
        
        iter_count += 1
    
    # Display policy and value function
    for state in range(NS):
        print(f"Epsilon-optimal action for state {state}={policy[state]}")
    
    for state in range(NS):
        print(f"The value function for state {state}={value[state]}")
    
    print(f"Average reward estimate= {sub_factor}")
    print(f"The number of iterations needed to converge= {iter_count}")
    
    return policy, value, sub_factor, iter_count


def rvia_with_verbose(tpm, trm, epsilon, verbose=True):
    """
    Relative Value Iteration with optional verbose output
    
    Args:
        tpm: Transition probability matrix
        trm: Transition reward matrix
        epsilon: Convergence threshold
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
        result = rvia(tpm, trm, epsilon)
        return result
    finally:
        if not verbose:
            # Restore original print function
            builtins.print = original_print 