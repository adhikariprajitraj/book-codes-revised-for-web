"""
MDP Average Reward Value Iteration Package

This package implements Value Iteration algorithms for Average Reward Markov Decision Processes (MDPs).
It is a Python translation of the original C code from the mdp/avg/val directory.

Modules:
    - constants: Global constants and parameters
    - via: Standard Value Iteration Algorithm implementation
    - rvia: Relative Value Iteration Algorithm implementation
    - main: Main execution module with example MDP
"""

from .constants import NS, NA, SMALL
from .via import via, via_with_verbose
from .rvia import rvia, rvia_with_verbose
from .main import create_example_mdp, main

__version__ = "1.0.0"
__author__ = "Python Translation of Original C Code"

__all__ = [
    'NS', 'NA', 'SMALL',
    'via', 'via_with_verbose',
    'rvia', 'rvia_with_verbose',
    'create_example_mdp', 'main'
] 