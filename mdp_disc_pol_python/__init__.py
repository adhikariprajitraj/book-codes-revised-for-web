"""
MDP Discounted Policy Iteration Package

This package implements Policy Iteration for Discounted Markov Decision Processes (MDPs).
It is a Python translation of the original C code from the mdp/disc/pol directory.

Modules:
    - constants: Global constants and parameters
    - solver: Linear system solver for policy evaluation
    - pid: Policy Iteration Algorithm implementation
    - main: Main execution module with example MDP
"""

from .constants import NS, NA, SMALL
from .solver import solver, solver_gauss_jordan
from .pid import pid, pid_with_verbose
from .main import create_example_mdp, main

__version__ = "1.0.0"
__author__ = "Python Translation of Original C Code"

__all__ = [
    'NS', 'NA', 'SMALL',
    'solver', 'solver_gauss_jordan',
    'pid', 'pid_with_verbose',
    'create_example_mdp', 'main'
] 