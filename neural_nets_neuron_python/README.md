# Single Neuron Neural Network

This is a Python implementation of a single neuron (perceptron) with batch training. It is a direct translation of the original C code from the `neural-nets/neuron-newrnd` directory.

## Overview

The package implements a single neuron with:
- **Architecture**: 2 inputs → 1 output
- **Activation**: Linear (no activation function)
- **Training**: Batch gradient descent
- **Optimization**: Learning rate decay and best weight tracking
- **Bonus**: Perceptron training with step activation

## Mathematical Framework

### Network Architecture

The network has the following structure:
```
Input Layer (2) → Output Layer (1, linear)
```

### Forward Pass

**Linear Output:**
```
y = ∑ᵢ wᵢ xᵢ + b
```

### Gradient Descent

**Error Calculation:**
```
E = ∑(target - output)²
```

**Weight Updates:**
```
∂E/∂wᵢ = ∑(target - output) × xᵢ
∂E/∂b = ∑(target - output)
```

**Update Rule:**
```
wᵢ = wᵢ + learning_rate × ∂E/∂wᵢ
b = b + learning_rate × ∂E/∂b
```

### Perceptron Training

For binary classification with step activation:
```
y = 1 if ∑ᵢ wᵢ xᵢ + b > 0 else 0
```

**Update Rule (when misclassified):**
```
wᵢ = wᵢ + learning_rate × (target - prediction) × xᵢ
b = b + learning_rate × (target - prediction)
```

## Installation

1. Clone or download this directory
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Example

To run the example single neuron training:

```bash
python3 main.py
```

### Using as a Module

```python
import numpy as np
from neural_nets_neuron_python import (
    init_net_numpy, load_data_from_file, neuron_vectorized,
    simu_net_vectorized
)

# Initialize network
weights, bias_weight = init_net_numpy()

# Load data
input_values, target_values = load_data_from_file('input.dat')

# Train network
best_weights, best_bias_weight, best_sse = neuron_vectorized(
    weights, bias_weight, input_values, target_values)

# Evaluate on new input
test_input = np.array([0.2, 0.5])
output = simu_net_vectorized(np.array([test_input]), best_weights, best_bias_weight)
print(f"Output: {output[0]}")
```

### Training a Perceptron

```python
from neural_nets_neuron_python import create_and_data, train_perceptron

# Create AND gate data
input_values, target_values = create_and_data()

# Train perceptron
weights, bias_weight, converged = train_perceptron(
    np.zeros(2), 0.0, input_values, target_values)

if converged:
    print("Perceptron converged!")
    print(f"Weights: {weights}")
    print(f"Bias: {bias_weight}")
```

### Creating Your Own Data

```python
import numpy as np
from neural_nets_neuron_python import neuron_vectorized, init_net_numpy

# Create custom data
input_values = np.random.uniform(0, 1, (10, 2))
target_values = 2 * input_values[:, 0] + 3 * input_values[:, 1] + 0.5

# Initialize and train
weights, bias_weight = init_net_numpy()
best_weights, best_bias, best_sse = neuron_vectorized(
    weights, bias_weight, input_values, target_values)
```

## File Structure

- `constants.py`: Global constants (NO_INPUTS, DATA_SIZE, ITERMAX)
- `initialization.py`: Weight initialization functions
- `forward_pass.py`: Forward pass through the network
- `data_loader.py`: Data loading and preprocessing
- `neuron.py`: Single neuron training algorithms
- `main.py`: Main execution module
- `__init__.py`: Package initialization
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Algorithm Details

### Batch Gradient Descent

The algorithm trains the neuron using batch gradient descent:

1. **Forward Pass**: Compute outputs for all training examples
2. **Error Calculation**: Compute sum squared error (SSE)
3. **Gradient Computation**: Compute gradients for weights and bias
4. **Weight Update**: Update weights using gradient descent
5. **Learning Rate Decay**: Reduce learning rate over time
6. **Best Weight Tracking**: Save weights with lowest SSE

### Perceptron Training

For binary classification tasks:

1. **Forward Pass**: Compute prediction using step activation
2. **Error Check**: Identify misclassified examples
3. **Weight Update**: Update weights only for misclassified examples
4. **Convergence Check**: Stop when all examples are correctly classified

### Training Parameters

- **Learning Rate**: Starts at 0.01, decays by factor of 0.99999 per iteration
- **Max Iterations**: 700 (configurable in constants.py)
- **Weight Initialization**: Uniform random in [-0.05, 0.05]
- **Convergence**: Based on SSE improvement or perfect classification

## Example Output

When you run `python3 main.py`, you should see output similar to:

```
Single Neuron Neural Network Training
Python Implementation

Initializing neural network...
Loading training data...
Training single neuron...
==================================================
Iteration 0, SSE: 91.884058
Iteration 100, SSE: 1.301236
Iteration 200, SSE: 0.537584
Iteration 300, SSE: 0.228889
Iteration 400, SSE: 0.098444
Iteration 500, SSE: 0.042497
Iteration 600, SSE: 0.018377
==================================================
Training completed!
Final SSE: 0.008022

==================================================
NEURAL NETWORK INFORMATION
==================================================
Architecture: 2 inputs -> 1 output (linear)

Input Weights:
[1.90911342 2.92935591]

Bias Weight:
0.575315234011511
==================================================

==================================================
PERCEPTRON TEST ON AND GATE
==================================================
Perceptron converged after 3 iterations
Perceptron successfully learned AND gate!
Final weights: [0.015 0.013]
Final bias: -0.02
```

## Key Features

- **Faithful Translation**: Direct translation of original C code
- **Vectorized Operations**: Both looped and vectorized implementations
- **Flexible Data**: Uses input.dat if available, otherwise generates synthetic data
- **Multiple Training Modes**: Linear regression and perceptron training
- **Comprehensive Evaluation**: Tests on training data and new inputs
- **Best Weight Tracking**: Saves weights with lowest error
- **Learning Rate Decay**: Adaptive learning rate for better convergence

## Comparison with Batch Neural Network

| Aspect | Single Neuron | Batch Neural Network |
|--------|---------------|---------------------|
| **Architecture** | 2→1 direct mapping | 2→3→1 with hidden layer |
| **Activation** | Linear only | Sigmoid hidden, linear output |
| **Complexity** | Linear functions only | Can learn non-linear functions |
| **Training** | Gradient descent | Backpropagation |
| **Use Case** | Linear regression | Complex pattern recognition |
| **Convergence** | Fast for linear problems | Slower but more powerful |

## Applications

### Linear Regression
The single neuron can learn linear relationships:
```
y = w₁x₁ + w₂x₂ + b
```

### Binary Classification (Perceptron)
With step activation, it can learn linearly separable patterns:
- AND gate
- OR gate
- NAND gate
- Linearly separable datasets

### Limitations
- Cannot learn XOR or other non-linearly separable patterns
- Limited to linear decision boundaries
- Requires linearly separable data for classification

## Dependencies

- `numpy`: For numerical computations and array operations

## License

This is a translation of the original C code. Please refer to the original code's license terms. 