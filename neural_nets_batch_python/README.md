# Batch Neural Network

This is a Python implementation of a feedforward neural network with batch backpropagation training. It is a direct translation of the original C code from the `neural-nets/batch-newrnd` directory.

## Overview

The package implements a feedforward neural network with:
- **Architecture**: 2 inputs → 3 hidden units → 1 output
- **Hidden Layer**: Sigmoid activation function
- **Output Layer**: Linear activation (no sigmoid)
- **Training**: Batch backpropagation with gradient descent
- **Optimization**: Learning rate decay and best weight tracking

## Mathematical Framework

### Network Architecture

The network has the following structure:
```
Input Layer (2) → Hidden Layer (3, sigmoid) → Output Layer (1, linear)
```

### Forward Pass

**Hidden Layer:**
```
h_j = σ(∑ᵢ wᵢⱼ xᵢ + bⱼ)
```
where σ is the sigmoid function: `σ(x) = 1 / (1 + e⁻ˣ)`

**Output Layer:**
```
y = ∑ⱼ vⱼ hⱼ + b_out
```

### Backpropagation

**Output Layer Gradients:**
```
δ_out = target - output
∂E/∂vⱼ = δ_out × hⱼ
∂E/∂b_out = δ_out
```

**Hidden Layer Gradients:**
```
δⱼ = δ_out × vⱼ × hⱼ × (1 - hⱼ)
∂E/∂wᵢⱼ = δⱼ × xᵢ
∂E/∂bⱼ = δⱼ
```

## Installation

1. Clone or download this directory
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Example

To run the example neural network training:

```bash
python3 main.py
```

### Using as a Module

```python
import numpy as np
from neural_nets_batch_python import (
    init_net_numpy, load_data_from_file, backprop_batch_vectorized,
    evaluator_vectorized
)

# Initialize network
ih_weights, ho_weights, bias_weight, vbias_weights = init_net_numpy()

# Load data
input_values, target_values = load_data_from_file('input.dat')

# Train network
best_ih_weights, best_ho_weights, best_bias_weight, best_vbias_weights, best_sse = \
    backprop_batch_vectorized(ih_weights, ho_weights, bias_weight, vbias_weights,
                             input_values, target_values)

# Evaluate on new input
test_input = np.array([0.2, 0.5])
output = evaluator_vectorized(best_ih_weights, best_ho_weights, 
                             best_bias_weight, best_vbias_weights, test_input)
```

### Creating Your Own Data

```python
import numpy as np
from neural_nets_batch_python import backprop_batch_vectorized, init_net_numpy

# Create custom data
input_values = np.random.uniform(0, 1, (10, 2))
target_values = 2 * input_values[:, 0] + 3 * input_values[:, 1] + 0.5

# Initialize and train
ih_weights, ho_weights, bias_weight, vbias_weights = init_net_numpy()
best_weights = backprop_batch_vectorized(ih_weights, ho_weights, bias_weight, 
                                        vbias_weights, input_values, target_values)
```

## File Structure

- `constants.py`: Global constants (NO_INPUTS, NO_HIDDEN, DATA_SIZE, ITERMAX)
- `activation.py`: Activation functions (sigmoid, sigmoid_derivative)
- `initialization.py`: Weight initialization functions
- `forward_pass.py`: Forward pass through the network
- `data_loader.py`: Data loading and preprocessing
- `backprop_batch.py`: Batch backpropagation training
- `evaluator.py`: Network evaluation functions
- `main.py`: Main execution module
- `__init__.py`: Package initialization
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Algorithm Details

### Batch Backpropagation

The algorithm trains the network using batch gradient descent:

1. **Forward Pass**: Compute outputs for all training examples
2. **Error Calculation**: Compute sum squared error (SSE)
3. **Backward Pass**: Compute gradients for all weights
4. **Weight Update**: Update weights using gradient descent
5. **Learning Rate Decay**: Reduce learning rate over time
6. **Best Weight Tracking**: Save weights with lowest SSE

### Training Parameters

- **Learning Rate**: Starts at 0.01, decays by factor of 0.99999 per iteration
- **Max Iterations**: 500 (configurable in constants.py)
- **Weight Initialization**: Uniform random in [-0.05, 0.05]
- **Convergence**: Based on SSE improvement

## Example Output

When you run `python3 main.py`, you should see output similar to:

```
Batch Neural Network Training
Python Implementation

Initializing neural network...
Loading training data...
Training neural network...
==================================================
Iteration 0, SSE: 10.732632
Iteration 100, SSE: 1.207053
Iteration 200, SSE: 1.185124
Iteration 300, SSE: 1.160279
Iteration 400, SSE: 1.130886
==================================================
Training completed!
Final SSE: 1.095874

==================================================
NEURAL NETWORK INFORMATION
==================================================
Architecture: 2 inputs -> 3 hidden -> 1 output

Input to Hidden Weights:
[[0.20428513 0.15094844 0.16360174]
 [0.27835701 0.28432707 0.18744836]]

Hidden to Output Weights:
[0.36520094 0.37852613 0.2791621 ]

Hidden Layer Bias Weights:
[ 0.0149674  0.0484923 -0.0482375]

Output Bias Weight:
0.35546521978253526
==================================================
```

## Key Features

- **Faithful Translation**: Direct translation of original C code
- **Vectorized Operations**: Both looped and vectorized implementations
- **Flexible Data**: Uses input.dat if available, otherwise generates synthetic data
- **Comprehensive Evaluation**: Tests on training data and new inputs
- **Best Weight Tracking**: Saves weights with lowest error
- **Learning Rate Decay**: Adaptive learning rate for better convergence

## Comparison with Single Neuron

| Aspect | Batch Neural Network | Single Neuron |
|--------|---------------------|---------------|
| **Architecture** | 2→3→1 with hidden layer | 2→1 direct mapping |
| **Activation** | Sigmoid hidden, linear output | Linear only |
| **Complexity** | Can learn non-linear functions | Linear functions only |
| **Training** | Backpropagation | Gradient descent |
| **Use Case** | Complex pattern recognition | Linear regression |

## Dependencies

- `numpy`: For numerical computations and array operations

## License

This is a translation of the original C code. Please refer to the original code's license terms. 