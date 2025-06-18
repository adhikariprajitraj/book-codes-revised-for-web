"""
Linear system solver for MDP Policy Iteration
Equivalent to solver.c in the original C code
"""

import numpy as np
from constants import NS


def solver(G):
    """
    Solves the linear system G[:, :-1] * x = G[:, -1] using numpy
    
    Args:
        G: Augmented matrix of shape (NS, NS+1)
        
    Returns:
        x: Solution vector of length NS
        
    Raises:
        SystemExit: If the matrix is singular
    """
    A = G[:, :-1]
    b = G[:, -1]
    
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError:
        print("Error in policy evaluation. Singular matrix.")
        exit(2)


def solver_gauss_jordan(G):
    """
    Alternative implementation using Gauss-Jordan elimination
    This is a direct translation of the original C solver.c
    """
    G = G.copy()
    x = np.zeros(NS)
    
    # Find max element and perform elimination
    for col in range(NS):
        pivot = -0.1
        pivot_row = col
        
        # Find the best pivot
        for row in range(col, NS):
            if abs(G[row, col]) > pivot:
                pivot = abs(G[row, col])
                pivot_row = row
        
        # Check if solution can be found
        if pivot <= 0.00001:
            print("Error in policy evaluation. Singular matrix.")
            exit(2)
        
        # Exchange rows to use the best pivot
        if pivot_row != col:
            G[col, :], G[pivot_row, :] = G[pivot_row, :].copy(), G[col, :].copy()
        
        # Do elimination
        for row1 in range(NS):
            if row1 != col:
                factor = G[row1, col] / G[col, col]
                for col2 in range(col, NS + 1):
                    G[row1, col2] -= factor * G[col, col2]
    
    # Find solution
    for row in range(NS):
        x[row] = G[row, NS] / G[row, row]
    
    return x 