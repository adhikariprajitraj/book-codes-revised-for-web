"""
Neural network evaluation functions
Equivalent to evaluator.c in the original C code
"""

import numpy as np
from constants import NO_INPUTS, NO_HIDDEN
from forward_pass import simu_net, simu_net_vectorized


def evaluator(ih_weights, ho_weights, bias_weight, vbias_weights, x):
    """
    Evaluate the neural network on a single input
    
    Args:
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
        x: Input vector (NO_INPUTS)
    """
    # Create single data point arrays
    input_values = np.zeros((1, NO_INPUTS))
    hidden_values = np.zeros((1, NO_HIDDEN))
    output_values = np.zeros(1)
    
    # Set input values
    for i in range(NO_INPUTS):
        input_values[0, i] = x[i]
    
    # Forward pass
    simu_net(input_values, ih_weights, ho_weights, bias_weight, vbias_weights,
             hidden_values, output_values)
    
    # Print results
    print(f"Input: {x}")
    print(f"Hidden layer outputs: {hidden_values[0]}")
    print(f"Network output: {output_values[0]}")
    
    return output_values[0]


def evaluator_vectorized(ih_weights, ho_weights, bias_weight, vbias_weights, x):
    """
    Vectorized evaluation of the neural network on a single input
    
    Args:
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
        x: Input vector (NO_INPUTS)
        
    Returns:
        float: Network output
    """
    # Reshape input for batch processing
    input_values = x.reshape(1, -1)
    
    # Forward pass
    hidden_values, output_values = simu_net_vectorized(
        input_values, ih_weights, ho_weights, bias_weight, vbias_weights)
    
    # Print results
    print(f"Input: {x}")
    print(f"Hidden layer outputs: {hidden_values[0]}")
    print(f"Network output: {output_values[0]}")
    
    return output_values[0]


def evaluate_multiple_inputs(ih_weights, ho_weights, bias_weight, vbias_weights, inputs):
    """
    Evaluate the neural network on multiple inputs
    
    Args:
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
        inputs: Input matrix (n_samples x NO_INPUTS)
        
    Returns:
        array: Network outputs
    """
    # Forward pass
    hidden_values, output_values = simu_net_vectorized(
        inputs, ih_weights, ho_weights, bias_weight, vbias_weights)
    
    return output_values


def print_network_info(ih_weights, ho_weights, bias_weight, vbias_weights):
    """
    Print information about the neural network
    
    Args:
        ih_weights: Input to hidden weights
        ho_weights: Hidden to output weights
        bias_weight: Output bias weight
        vbias_weights: Hidden layer bias weights
    """
    print("=" * 50)
    print("NEURAL NETWORK INFORMATION")
    print("=" * 50)
    print(f"Architecture: {NO_INPUTS} inputs -> {NO_HIDDEN} hidden -> 1 output")
    print()
    
    print("Input to Hidden Weights:")
    print(ih_weights)
    print()
    
    print("Hidden to Output Weights:")
    print(ho_weights)
    print()
    
    print("Hidden Layer Bias Weights:")
    print(vbias_weights)
    print()
    
    print("Output Bias Weight:")
    print(bias_weight)
    print("=" * 50) 