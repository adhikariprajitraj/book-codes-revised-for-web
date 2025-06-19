"""
Single Neuron Forward Pass (simulation)
Equivalent to simu_net.c in the original C code
"""

import numpy as np
from constants import NO_INPUTS, DATA_SIZE


def simu_net(input_values, weights, bias_weight, output_values):
    """
    Forward pass through the single neuron
    
    Args:
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        weights: Input weights (NO_INPUTS)
        bias_weight: Bias weight (scalar)
        output_values: Network outputs (DATA_SIZE) - output
    """
    # Evaluates the values in the net
    # Linear activation function (no sigmoid for single neuron)
    
    for p in range(DATA_SIZE):
        # For each data point
        sum_val = 0.0
        for i in range(NO_INPUTS):
            # Over all inputs
            sum_val += weights[i] * input_values[p, i]
        
        sum_val += bias_weight
        output_values[p] = sum_val


def simu_net_vectorized(input_values, weights, bias_weight):
    """
    Vectorized forward pass through the single neuron
    
    Args:
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        weights: Input weights (NO_INPUTS)
        bias_weight: Bias weight (scalar)
        
    Returns:
        array: Network outputs
    """
    # Linear combination of inputs with bias
    output_values = np.dot(input_values, weights) + bias_weight
    
    return output_values


def simu_net_with_activation(input_values, weights, bias_weight, activation='linear'):
    """
    Forward pass with optional activation function
    
    Args:
        input_values: Input data (DATA_SIZE x NO_INPUTS)
        weights: Input weights (NO_INPUTS)
        bias_weight: Bias weight (scalar)
        activation: Activation function ('linear', 'sigmoid', 'step')
        
    Returns:
        array: Network outputs
    """
    # Linear combination
    output_values = np.dot(input_values, weights) + bias_weight
    
    # Apply activation function
    if activation == 'sigmoid':
        output_values = 1.0 / (1.0 + np.exp(-np.clip(output_values, -709.0, 709.0)))
    elif activation == 'step':
        output_values = (output_values > 0).astype(float)
    # 'linear' activation is just the identity function
    
    return output_values 