"""
Data loading functions for neural networks
Equivalent to reader.c in the original C code
"""

import numpy as np
from constants import NO_INPUTS, DATA_SIZE


def reader(input_values, target_values):
    """
    Read training data from input.dat file
    
    Args:
        input_values: Input data array (DATA_SIZE x NO_INPUTS) - output
        target_values: Target values array (DATA_SIZE) - output
    """
    try:
        with open('input.dat', 'r') as file:
            data = []
            for line in file:
                line = line.strip()
                if line:
                    data.append(float(line))
        
        # Reshape data into input and target arrays
        for p in range(DATA_SIZE):
            for i in range(NO_INPUTS):
                input_values[p, i] = data[p * 3 + i]
            target_values[p] = data[p * 3 + 2]
            
    except FileNotFoundError:
        print("Warning: input.dat not found. Using default data.")
        create_default_data(input_values, target_values)


def create_default_data(input_values, target_values):
    """
    Create default training data if input.dat is not available
    
    Args:
        input_values: Input data array (DATA_SIZE x NO_INPUTS) - output
        target_values: Target values array (DATA_SIZE) - output
    """
    # Create some sample data for XOR-like problem
    np.random.seed(42)
    
    for p in range(DATA_SIZE):
        # Generate random inputs between 0 and 1
        input_values[p, 0] = np.random.random()
        input_values[p, 1] = np.random.random()
        
        # Create a simple target function (e.g., sum of inputs)
        target_values[p] = input_values[p, 0] + input_values[p, 1]


def load_data_from_file(filename='input.dat'):
    """
    Load data from file and return as numpy arrays
    
    Args:
        filename: Name of the data file
        
    Returns:
        tuple: (input_values, target_values)
    """
    input_values = np.zeros((DATA_SIZE, NO_INPUTS))
    target_values = np.zeros(DATA_SIZE)
    
    try:
        with open(filename, 'r') as file:
            data = []
            for line in file:
                line = line.strip()
                if line:
                    data.append(float(line))
        
        # Reshape data into input and target arrays
        for p in range(DATA_SIZE):
            for i in range(NO_INPUTS):
                input_values[p, i] = data[p * 3 + i]
            target_values[p] = data[p * 3 + 2]
            
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default data.")
        create_default_data(input_values, target_values)
    
    return input_values, target_values


def create_xor_data():
    """
    Create XOR-like training data
    
    Returns:
        tuple: (input_values, target_values)
    """
    input_values = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.3],
        [0.3, 0.5]
    ])
    
    target_values = np.array([
        0, 1, 1, 0, 0, 1, 1, 0, 0.8, 0.8
    ])
    
    return input_values, target_values 