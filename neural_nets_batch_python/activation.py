"""
Activation functions for neural networks
Equivalent to compute_sigmoid.c in the original C code
"""

import numpy as np


def compute_sigmoid(input_val):
    """
    Sigmoid activation function
    
    Args:
        input_val: Input value
        
    Returns:
        Output of sigmoid function
    """
    if input_val < -709.0:
        return 1.0
    elif input_val > 709.0:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-input_val))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid function
    
    Args:
        x: Input value (should be sigmoid output)
        
    Returns:
        Derivative of sigmoid at x
    """
    return x * (1 - x)


def sigmoid_vectorized(x):
    """
    Vectorized sigmoid function for numpy arrays
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid output array
    """
    # Clip to avoid overflow
    x = np.clip(x, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-x)) 