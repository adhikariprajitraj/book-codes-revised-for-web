"""
Single Neuron Training Algorithm
Equivalent to neuron.c in the original C code
"""

import numpy as np
from constants import NO_INPUTS, DATA_SIZE, ITERMAX
from forward_pass import simu_net


def neuron(weights, bias_weight, input_values, target_values):
    """
    Train a single neuron in batch mode using backpropagation
    
    Args:
        weights: Input weights (NO_INPUTS)
        bias_weight: Bias weight (scalar)
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        target_values: Target values (DATA_SIZE)
    """
    # This function trains a neuron net in batch mode
    # A bias unit has been used
    
    learning_rate = 0.01
    best_sse = 100000  # Some large number
    
    # Initialize arrays for storing intermediate values
    output_values = np.zeros(DATA_SIZE)
    output_deltas = np.zeros(DATA_SIZE)
    
    # Arrays to store best weights
    best_weights = np.zeros(NO_INPUTS)
    best_bias_weight = 0.0
    
    # Initial forward pass
    simu_net(input_values, weights, bias_weight, output_values)
    
    iter_count = 0
    
    while iter_count < ITERMAX:
        # One iteration of training
        
        # Update bias weight
        change = 0
        for p in range(DATA_SIZE):
            # Computing deltas for output unit for each point
            output_deltas[p] = target_values[p] - output_values[p]
            change += output_deltas[p]
        bias_weight[0] = bias_weight[0] + (learning_rate * change)
        
        # Updating weights from input units to output
        for i in range(NO_INPUTS):
            change = 0
            for p in range(DATA_SIZE):
                # Sum change over all points
                change += output_deltas[p] * input_values[p, i]
            weights[i] = weights[i] + (change * learning_rate)
        
        # Forward pass with updated weights
        simu_net(input_values, weights, bias_weight, output_values)
        
        # Calculate SSE
        sse = 0
        for p in range(DATA_SIZE):
            # Finding sum of the SSE
            error = output_values[p] - target_values[p]
            sse += (error * error)
        
        print(f"# of training iterations={iter_count}")
        
        if sse < best_sse:
            # Save the weights as the best weights so far
            best_sse = sse
            for i in range(NO_INPUTS):
                best_weights[i] = weights[i]
            best_bias_weight = bias_weight[0]
        
        print(f"SSE={best_sse}")
        iter_count += 1
        learning_rate = learning_rate * 0.99999
    
    # Return the best weights learned
    for i in range(NO_INPUTS):
        weights[i] = best_weights[i]
    bias_weight[0] = best_bias_weight


def neuron_vectorized(weights, bias_weight, input_values, target_values, max_iterations=ITERMAX):
    """
    Vectorized version of single neuron training
    
    Args:
        weights: Input weights (NO_INPUTS)
        bias_weight: Bias weight (scalar)
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        target_values: Target values (DATA_SIZE)
        max_iterations: Maximum number of training iterations
        
    Returns:
        tuple: (best_weights, best_bias_weight, best_sse)
    """
    learning_rate = 0.01
    best_sse = float('inf')
    
    # Store best weights
    best_weights = weights.copy()
    best_bias_weight = bias_weight
    
    for iter_count in range(max_iterations):
        # Forward pass
        output_values = simu_net_vectorized(input_values, weights, bias_weight)
        
        # Output deltas
        output_deltas = target_values - output_values
        
        # Update bias weight
        bias_weight += learning_rate * np.sum(output_deltas)
        
        # Update input weights
        weights += learning_rate * np.dot(input_values.T, output_deltas)
        
        # Calculate SSE
        sse = np.sum((output_values - target_values) ** 2)
        
        if sse < best_sse:
            best_sse = sse
            best_weights = weights.copy()
            best_bias_weight = bias_weight
        
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}, SSE: {best_sse:.6f}")
        
        learning_rate *= 0.99999
    
    return best_weights, best_bias_weight, best_sse


def train_perceptron(weights, bias_weight, input_values, target_values, max_iterations=ITERMAX):
    """
    Train a perceptron (single neuron with step activation)
    
    Args:
        weights: Input weights (NO_INPUTS)
        bias_weight: Bias weight (scalar)
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        target_values: Target values (DATA_SIZE) - should be binary {0, 1}
        max_iterations: Maximum number of training iterations
        
    Returns:
        tuple: (best_weights, best_bias_weight, converged)
    """
    learning_rate = 0.01
    converged = False
    
    for iter_count in range(max_iterations):
        misclassified = 0
        
        for p in range(DATA_SIZE):
            # Forward pass
            output = np.dot(weights, input_values[p]) + bias_weight
            
            # Step activation
            prediction = 1 if output > 0 else 0
            
            # Update if misclassified
            if prediction != target_values[p]:
                misclassified += 1
                error = target_values[p] - prediction
                
                # Update weights
                weights += learning_rate * error * input_values[p]
                bias_weight += learning_rate * error
        
        # Check convergence
        if misclassified == 0:
            converged = True
            print(f"Perceptron converged after {iter_count + 1} iterations")
            break
        
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}, Misclassified: {misclassified}")
    
    if not converged:
        print(f"Perceptron did not converge after {max_iterations} iterations")
    
    return weights, bias_weight, converged 