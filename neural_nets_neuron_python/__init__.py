"""
Single Neuron Neural Network Package

This package implements a single neuron (perceptron) with batch training.
It is a Python translation of the original C code from the neural-nets/neuron-newrnd directory.

Modules:
    - constants: Global constants and network architecture
    - initialization: Weight initialization functions
    - forward_pass: Forward pass through the network
    - data_loader: Data loading and preprocessing
    - neuron: Single neuron training algorithms
    - main: Main execution module
"""

from .constants import NO_INPUTS, DATA_SIZE, ITERMAX
from .initialization import init_net, init_net_numpy, init_weights_zero, init_weights_random
from .forward_pass import simu_net, simu_net_vectorized, simu_net_with_activation
from .data_loader import reader, load_data_from_file, create_linear_data, create_and_data, create_or_data
from .neuron import neuron, neuron_vectorized, train_perceptron
from .main import main

__version__ = "1.0.0"
__author__ = "Python Translation of Original C Code"

__all__ = [
    'NO_INPUTS', 'DATA_SIZE', 'ITERMAX',
    'init_net', 'init_net_numpy', 'init_weights_zero', 'init_weights_random',
    'simu_net', 'simu_net_vectorized', 'simu_net_with_activation',
    'reader', 'load_data_from_file', 'create_linear_data', 'create_and_data', 'create_or_data',
    'neuron', 'neuron_vectorized', 'train_perceptron',
    'main'
] 