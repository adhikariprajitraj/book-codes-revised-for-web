"""
Main module for Single Neuron Neural Network
Equivalent to main.c in the original C code
"""

import numpy as np
from initialization import init_net_numpy
from data_loader import load_data_from_file, create_linear_data, create_and_data, create_or_data
from neuron import neuron_vectorized, train_perceptron
from forward_pass import simu_net_vectorized
from constants import NO_INPUTS, DATA_SIZE


def main():
    """
    Main function that trains and evaluates the single neuron
    """
    print("Single Neuron Neural Network Training")
    print("Python Implementation")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(10)
    
    # Initialize network weights
    print("Initializing neural network...")
    weights, bias_weight = init_net_numpy()
    
    # Load training data
    print("Loading training data...")
    try:
        input_values, target_values = load_data_from_file('input.dat')
        print("Loaded data from input.dat")
    except FileNotFoundError:
        print("Using linear regression training data")
        input_values, target_values = create_linear_data()
    
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
    print("Training single neuron...")
    print("=" * 50)
    
    best_weights, best_bias_weight, best_sse = neuron_vectorized(
        weights, bias_weight, input_values, target_values)
    
    print("=" * 50)
    print("Training completed!")
    print(f"Final SSE: {best_sse:.6f}")
    print()
    
    # Print network information
    print("=" * 50)
    print("NEURAL NETWORK INFORMATION")
    print("=" * 50)
    print(f"Architecture: {NO_INPUTS} inputs -> 1 output (linear)")
    print()
    print("Input Weights:")
    print(best_weights)
    print()
    print("Bias Weight:")
    print(best_bias_weight)
    print("=" * 50)
    
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
        output = simu_net_vectorized(np.array([x]), best_weights, best_bias_weight)
        print(f"Input: {x}")
        print(f"Network output: {output[0]}")
    
    # Evaluate on training data
    print("\n" + "=" * 50)
    print("EVALUATION ON TRAINING DATA")
    print("=" * 50)
    
    predictions = simu_net_vectorized(input_values, best_weights, best_bias_weight)
    
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
    
    # Test perceptron on AND gate
    print("\n" + "=" * 50)
    print("PERCEPTRON TEST ON AND GATE")
    print("=" * 50)
    
    and_inputs, and_targets = create_and_data()
    p_weights, p_bias, converged = train_perceptron(
        np.zeros(NO_INPUTS), 0.0, and_inputs, and_targets, max_iterations=1000)
    
    if converged:
        print("Perceptron successfully learned AND gate!")
        print(f"Final weights: {p_weights}")
        print(f"Final bias: {p_bias}")
        
        # Test AND gate
        for i in range(len(and_inputs)):
            output = simu_net_vectorized(np.array([and_inputs[i]]), p_weights, p_bias)
            prediction = 1 if output[0] > 0 else 0
            print(f"  Input: {and_inputs[i]}, Target: {and_targets[i]}, "
                  f"Prediction: {prediction}, Raw output: {output[0]:.3f}")


if __name__ == "__main__":
    main() 