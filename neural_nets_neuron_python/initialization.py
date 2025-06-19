"""
Single Neuron Initialization Functions
"""

import numpy as np
from constants import NO_INPUTS


def init_net(weights, bias_weight):
    """
    Initialize single neuron weights
    
    Args:
        weights: Input weights (NO_INPUTS)
        bias_weight: Bias weight (scalar)
    """
    # Initialize input weights with small random values
    for i in range(NO_INPUTS):
        weights[i] = (np.random.random() - 0.5) * 0.1
    
    # Initialize bias weight with small random value
    bias_weight[0] = (np.random.random() - 0.5) * 0.1


def init_net_numpy():
    """
    Initialize single neuron weights using numpy arrays
    
    Returns:
        tuple: (weights, bias_weight)
    """
    # Initialize input weights
    weights = np.random.uniform(-0.05, 0.05, NO_INPUTS)
    
    # Initialize bias weight
    bias_weight = np.random.uniform(-0.05, 0.05, 1)[0]
    
    return weights, bias_weight


def init_weights_zero():
    """
    Initialize weights to zero
    
    Returns:
        tuple: (weights, bias_weight)
    """
    weights = np.zeros(NO_INPUTS)
    bias_weight = 0.0
    
    return weights, bias_weight


def init_weights_random(scale=0.1):
    """
    Initialize weights with random values
    
    Args:
        scale: Scale factor for random initialization
        
    Returns:
        tuple: (weights, bias_weight)
    """
    weights = np.random.uniform(-scale, scale, NO_INPUTS)
    bias_weight = np.random.uniform(-scale, scale, 1)[0]
    
    return weights, bias_weight 