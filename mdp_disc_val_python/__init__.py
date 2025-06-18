"""
MDP Discounted Value Iteration Package

This package implements Value Iteration algorithms for Discounted Markov Decision Processes (MDPs).
It is a Python translation of the original C code from the mdp/disc/val directory.

Modules:
    - constants: Global constants and parameters
    - vi: Standard Value Iteration Algorithm implementation
    - gsvi: Gauss-Seidel Value Iteration Algorithm implementation
    - rvi: Relative Value Iteration Algorithm implementation
    - main: Main execution module with example MDP
"""

from .constants import NS, NA, SMALL
from .vi import vi, vi_with_verbose
from .gsvi import gsvi, gsvi_with_verbose
from .rvi import rvi, rvi_with_verbose
from .main import create_example_mdp, main

__version__ = "1.0.0"
__author__ = "Python Translation of Original C Code"

__all__ = [
    'NS', 'NA', 'SMALL',
    'vi', 'vi_with_verbose',
    'gsvi', 'gsvi_with_verbose',
    'rvi', 'rvi_with_verbose',
    'create_example_mdp', 'main'
] 