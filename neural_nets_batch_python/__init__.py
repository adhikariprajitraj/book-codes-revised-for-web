"""
Batch Neural Network Package

This package implements a feedforward neural network with batch backpropagation training.
It is a Python translation of the original C code from the neural-nets/batch-newrnd directory.

Modules:
    - constants: Global constants and network architecture
    - activation: Activation functions (sigmoid)
    - initialization: Weight initialization functions
    - forward_pass: Forward pass through the network
    - data_loader: Data loading and preprocessing
    - backprop_batch: Batch backpropagation training
    - evaluator: Network evaluation functions
    - main: Main execution module
"""

from .constants import NO_INPUTS, NO_HIDDEN, DATA_SIZE, ITERMAX
from .activation import compute_sigmoid, sigmoid_derivative, sigmoid_vectorized
from .initialization import init_net, init_net_numpy
from .forward_pass import simu_net, simu_net_vectorized
from .data_loader import reader, load_data_from_file, create_xor_data
from .backprop_batch import backprop_batch, backprop_batch_vectorized
from .evaluator import evaluator, evaluator_vectorized, print_network_info
from .main import main

__version__ = "1.0.0"
__author__ = "Python Translation of Original C Code"

__all__ = [
    'NO_INPUTS', 'NO_HIDDEN', 'DATA_SIZE', 'ITERMAX',
    'compute_sigmoid', 'sigmoid_derivative', 'sigmoid_vectorized',
    'init_net', 'init_net_numpy',
    'simu_net', 'simu_net_vectorized',
    'reader', 'load_data_from_file', 'create_xor_data',
    'backprop_batch', 'backprop_batch_vectorized',
    'evaluator', 'evaluator_vectorized', 'print_network_info',
    'main'
] 