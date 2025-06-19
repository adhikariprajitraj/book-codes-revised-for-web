"""
Neural network initialization functions
Equivalent to init_net.c in the original C code
"""

import numpy as np
from constants import NO_INPUTS, NO_HIDDEN


def init_net(ih_weights, ho_weights, bias_weight, vbias_weights):
    """
    Initialize neural network weights
    
    Args:
        ih_weights: Input to hidden weights (NO_INPUTS x NO_HIDDEN)
        ho_weights: Hidden to output weights (NO_HIDDEN)
        bias_weight: Output bias weight (scalar)
        vbias_weights: Hidden layer bias weights (NO_HIDDEN)
    """
    # Initialize input to hidden weights with small random values
    for i in range(NO_INPUTS):
        for h in range(NO_HIDDEN):
            ih_weights[i, h] = (np.random.random() - 0.5) * 0.1
    
    # Initialize hidden to output weights with small random values
    for h in range(NO_HIDDEN):
        ho_weights[h] = (np.random.random() - 0.5) * 0.1
    
    # Initialize bias weights with small random values
    bias_weight[0] = (np.random.random() - 0.5) * 0.1
    
    for h in range(NO_HIDDEN):
        vbias_weights[h] = (np.random.random() - 0.5) * 0.1


def init_net_numpy():
    """
    Initialize neural network weights using numpy arrays
    
    Returns:
        tuple: (ih_weights, ho_weights, bias_weight, vbias_weights)
    """
    # Initialize input to hidden weights
    ih_weights = np.random.uniform(-0.05, 0.05, (NO_INPUTS, NO_HIDDEN))
    
    # Initialize hidden to output weights
    ho_weights = np.random.uniform(-0.05, 0.05, NO_HIDDEN)
    
    # Initialize bias weights
    bias_weight = np.random.uniform(-0.05, 0.05, 1)[0]
    vbias_weights = np.random.uniform(-0.05, 0.05, NO_HIDDEN)
    
    return ih_weights, ho_weights, bias_weight, vbias_weights 