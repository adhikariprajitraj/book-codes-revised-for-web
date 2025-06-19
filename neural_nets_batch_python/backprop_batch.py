"""
Batch Backpropagation Algorithm for Neural Networks
Equivalent to backprop_batch.c in the original C code
"""

import numpy as np
from constants import NO_INPUTS, NO_HIDDEN, DATA_SIZE, ITERMAX
from forward_pass import simu_net, simu_net_vectorized


def backprop_batch(ih_weights, ho_weights, bias_weight, vbias_weights,
                   input_values, target_values):
    """
    Train a neural net in batch mode using backpropagation
    
    Args:
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        target_values: Target values (DATA_SIZE)
    """
    # This function trains a neural net in batch mode
    # Uses a single hidden layer with sigmoid transfer function only from
    # input to hidden, not from hidden to output
    # A bias unit has been used
    
    learning_rate = 0.01
    best_sse = 100000  # Some large number
    
    # Initialize arrays for storing intermediate values
    output_values = np.zeros(DATA_SIZE)
    hidden_values = np.zeros((DATA_SIZE, NO_HIDDEN))
    hidden_deltas = np.zeros((DATA_SIZE, NO_HIDDEN))
    output_deltas = np.zeros(DATA_SIZE)
    
    # Arrays to store best weights
    best_ih_weights = np.zeros((NO_INPUTS, NO_HIDDEN))
    best_ho_weights = np.zeros(NO_HIDDEN)
    best_vbias_weights = np.zeros(NO_HIDDEN)
    best_bias_weight = 0.0
    
    # Initial forward pass
    simu_net(input_values, ih_weights, ho_weights, bias_weight, vbias_weights,
             hidden_values, output_values)
    
    iter_count = 0
    
    while iter_count < ITERMAX:
        # One iteration of training
        
        # Computing deltas for output unit for each point
        for p in range(DATA_SIZE):
            output_deltas[p] = target_values[p] - output_values[p]
        
        # Update bias weight
        change = 0
        for p in range(DATA_SIZE):
            # Summing over all points
            change += output_deltas[p]
        bias_weight[0] = bias_weight[0] + (learning_rate * change)
        
        # Compute hidden deltas
        for h in range(NO_HIDDEN):
            # Computing delta for hidden units for each point
            for p in range(DATA_SIZE):
                # For each point (training example)
                hidden_deltas[p, h] = (output_deltas[p] * ho_weights[h] *
                                      hidden_values[p, h] * (1 - hidden_values[p, h]))
        
        # Updating weights from input units to hidden units
        for i in range(NO_INPUTS):
            for h in range(NO_HIDDEN):
                change = 0
                for p in range(DATA_SIZE):
                    # Sum change over all points
                    change += hidden_deltas[p, h] * input_values[p, i]
                ih_weights[i, h] = ih_weights[i, h] + (change * learning_rate)
        
        # Updating virtual bias weights
        for h in range(NO_HIDDEN):
            change = 0
            for p in range(DATA_SIZE):
                # Sum change over all points
                change += hidden_deltas[p, h]
            vbias_weights[h] = vbias_weights[h] + (change * learning_rate)
        
        # Update weights from hidden to output units
        for h in range(NO_HIDDEN):
            change = 0
            for p in range(DATA_SIZE):
                # Sum over all points
                change += output_deltas[p] * hidden_values[p, h]
            ho_weights[h] = ho_weights[h] + (learning_rate * change)
        
        # Forward pass with updated weights
        simu_net(input_values, ih_weights, ho_weights, bias_weight, vbias_weights,
                 hidden_values, output_values)
        
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
            for h in range(NO_HIDDEN):
                best_ho_weights[h] = ho_weights[h]
                best_vbias_weights[h] = vbias_weights[h]
                for i in range(NO_INPUTS):
                    best_ih_weights[i, h] = ih_weights[i, h]
            best_bias_weight = bias_weight[0]
        
        print(f"SSE={best_sse}")
        iter_count += 1
        learning_rate = learning_rate * 0.99999
    
    # Return the best weights learned
    for h in range(NO_HIDDEN):
        ho_weights[h] = best_ho_weights[h]
        vbias_weights[h] = best_vbias_weights[h]
        for i in range(NO_INPUTS):
            ih_weights[i, h] = best_ih_weights[i, h]
    bias_weight[0] = best_bias_weight


def backprop_batch_vectorized(ih_weights, ho_weights, bias_weight, vbias_weights,
                             input_values, target_values, max_iterations=ITERMAX):
    """
    Vectorized version of batch backpropagation
    
    Args:
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        target_values: Target values (DATA_SIZE)
        max_iterations: Maximum number of training iterations
        
    Returns:
        tuple: (best_ih_weights, best_ho_weights, best_bias_weight, best_vbias_weights, best_sse)
    """
    learning_rate = 0.01
    best_sse = float('inf')
    
    # Store best weights
    best_ih_weights = ih_weights.copy()
    best_ho_weights = ho_weights.copy()
    best_bias_weight = bias_weight
    best_vbias_weights = vbias_weights.copy()
    
    for iter_count in range(max_iterations):
        # Forward pass
        hidden_values, output_values = simu_net_vectorized(
            input_values, ih_weights, ho_weights, bias_weight, vbias_weights)
        
        # Output deltas
        output_deltas = target_values - output_values
        
        # Update bias weight
        bias_weight += learning_rate * np.sum(output_deltas)
        
        # Hidden deltas
        hidden_deltas = (output_deltas.reshape(-1, 1) * ho_weights.reshape(1, -1) *
                        hidden_values * (1 - hidden_values))
        
        # Update input to hidden weights
        ih_weights += learning_rate * np.dot(input_values.T, hidden_deltas)
        
        # Update hidden bias weights
        vbias_weights += learning_rate * np.sum(hidden_deltas, axis=0)
        
        # Update hidden to output weights
        ho_weights += learning_rate * np.dot(hidden_values.T, output_deltas)
        
        # Calculate SSE
        sse = np.sum((output_values - target_values) ** 2)
        
        if sse < best_sse:
            best_sse = sse
            best_ih_weights = ih_weights.copy()
            best_ho_weights = ho_weights.copy()
            best_bias_weight = bias_weight
            best_vbias_weights = vbias_weights.copy()
        
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}, SSE: {best_sse:.6f}")
        
        learning_rate *= 0.99999
    
    return best_ih_weights, best_ho_weights, best_bias_weight, best_vbias_weights, best_sse 