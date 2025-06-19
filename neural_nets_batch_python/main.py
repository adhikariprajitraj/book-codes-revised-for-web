"""
Main module for Batch Neural Network
Equivalent to main.c in the original C code
"""

import numpy as np
from initialization import init_net_numpy
from data_loader import load_data_from_file, create_xor_data
from backprop_batch import backprop_batch_vectorized
from evaluator import evaluator_vectorized, print_network_info
from constants import NO_INPUTS, NO_HIDDEN, DATA_SIZE


def main():
    """
    Main function that trains and evaluates the neural network
    """
    print("Batch Neural Network Training")
    print("Python Implementation")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(10)
    
    # Initialize network weights
    print("Initializing neural network...")
    ih_weights, ho_weights, bias_weight, vbias_weights = init_net_numpy()
    
    # Load training data
    print("Loading training data...")
    try:
        input_values, target_values = load_data_from_file('input.dat')
        print("Loaded data from input.dat")
    except FileNotFoundError:
        print("Using XOR-like training data")
        input_values, target_values = create_xor_data()
    
    # Print data information
    print(f"Training data shape: {input_values.shape}")
    print(f"Target values shape: {target_values.shape}")
    print()
    
    # Print sample data
    print("Sample training data:")
    for i in range(min(5, DATA_SIZE)):
        print(f"  Input: {input_values[i]}, Target: {target_values[i]}")
    print()
    
    # Train the network
    print("Training neural network...")
    print("=" * 50)
    
    best_ih_weights, best_ho_weights, best_bias_weight, best_vbias_weights, best_sse = \
        backprop_batch_vectorized(ih_weights, ho_weights, bias_weight, vbias_weights,
                                 input_values, target_values)
    
    print("=" * 50)
    print("Training completed!")
    print(f"Final SSE: {best_sse:.6f}")
    print()
    
    # Print network information
    print_network_info(best_ih_weights, best_ho_weights, best_bias_weight, best_vbias_weights)
    
    # Evaluate on test inputs
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST INPUTS")
    print("=" * 50)
    
    # Test inputs from original C code
    test_inputs = [
        np.array([0.2, 0.5]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5])
    ]
    
    for i, x in enumerate(test_inputs):
        print(f"\nTest Input {i+1}:")
        output = evaluator_vectorized(best_ih_weights, best_ho_weights, 
                                     best_bias_weight, best_vbias_weights, x)
    
    # Evaluate on training data
    print("\n" + "=" * 50)
    print("EVALUATION ON TRAINING DATA")
    print("=" * 50)
    
    from forward_pass import simu_net_vectorized
    hidden_values, predictions = simu_net_vectorized(
        input_values, best_ih_weights, best_ho_weights, best_bias_weight, best_vbias_weights)
    
    print("Training Results:")
    for i in range(DATA_SIZE):
        print(f"  Input: {input_values[i]}, Target: {target_values[i]:.3f}, "
              f"Prediction: {predictions[i]:.3f}, Error: {abs(predictions[i] - target_values[i]):.3f}")
    
    # Calculate final metrics
    mse = np.mean((predictions - target_values) ** 2)
    mae = np.mean(np.abs(predictions - target_values))
    
    print(f"\nFinal Metrics:")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Sum Squared Error: {best_sse:.6f}")


if __name__ == "__main__":
    main() 