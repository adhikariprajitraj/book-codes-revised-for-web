"""
Neural network forward pass (simulation)
Equivalent to simu_net.c in the original C code
"""

import numpy as np
from constants import NO_INPUTS, NO_HIDDEN, DATA_SIZE
from activation import compute_sigmoid


def simu_net(input_values, ih_weights, ho_weights, bias_weight, vbias_weights,
             hidden_values, output_values):
    """
    Forward pass through the neural network
    
    Args:
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
        hidden_values: Hidden layer outputs (DATA_SIZE x NO_HIDDEN) - output
        output_values: Network outputs (DATA_SIZE) - output
    """
    # Evaluate the values in the net
    # Sigmoid thresholding is done only from input to hidden layer
    # but not from hidden to output layer
    
    for p in range(DATA_SIZE):
        # For each data point
        for h in range(NO_HIDDEN):
            # Calculate the sum of all inputs to each hidden node
            sum1 = 0.0
            for i in range(NO_INPUTS):
                # Over all inputs
                sum1 += ih_weights[i, h] * input_values[p, i]
            
            sum1 += vbias_weights[h]
            hidden_values[p, h] = compute_sigmoid(sum1)
    
    for p in range(DATA_SIZE):
        sum2 = 0
        for h in range(NO_HIDDEN):
            # To calculate output; notice this output is not thresholded
            # with any sigmoid unit
            sum2 += ho_weights[h] * hidden_values[p, h]
        
        sum2 += bias_weight
        output_values[p] = sum2


def simu_net_vectorized(input_values, ih_weights, ho_weights, bias_weight, vbias_weights):
    """
    Vectorized forward pass through the neural network
    
    Args:
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
        
    Returns:
        tuple: (hidden_values, output_values)
    """
    # Hidden layer computation with sigmoid activation
    hidden_inputs = np.dot(input_values, ih_weights) + vbias_weights
    hidden_values = 1.0 / (1.0 + np.exp(-np.clip(hidden_inputs, -709.0, 709.0)))
    
    # Output layer computation (linear activation)
    output_values = np.dot(hidden_values, ho_weights) + bias_weight
    
    return hidden_values, output_values 