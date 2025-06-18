# MDP Discounted Value Iteration

This is a Python implementation of Value Iteration algorithms for Discounted Markov Decision Processes (MDPs). It is a direct translation of the original C code from the `mdp/disc/val` directory.

## Overview

The package implements three Value Iteration algorithms for solving MDPs with the discounted reward criterion:

1. **Standard Value Iteration (VI)**: Classic value iteration algorithm
2. **Gauss-Seidel Value Iteration (GSVI)**: In-place updates for faster convergence
3. **Relative Value Iteration (RVI)**: Bounded values using reference state subtraction

## Mathematical Framework

### Discounted MDP Objective

For discounted MDPs, the objective is to maximize the discounted sum of future rewards:

$$\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) \right]$$

where $\gamma \in [0,1)$ is the discount factor.

### Value Iteration Algorithm

**Value Iteration** directly updates the value function:

$$V_{k+1}(s) = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]$$

### Convergence Criterion

The algorithm terminates when the maximum change in value function is below a threshold:

$$\max_s |V_{k+1}(s) - V_k(s)| < \epsilon$$

The termination factor is adjusted as: $\epsilon = \epsilon \cdot (1-\gamma) \cdot 0.5 / \gamma$

## Installation

1. Clone or download this directory
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Example

To run the example MDP that comes with the original C code:

```bash
python3 main.py
```

### Using as a Module

```python
import numpy as np
from mdp_disc_val_python import vi, gsvi, rvi, create_example_mdp

# Create the example MDP
tpm, trm = create_example_mdp()

# Set parameters
d_factor = 0.8
epsilon = 0.001

# Run different value iteration algorithms
policy_vi, values_vi, iterations_vi = vi(tpm, trm, d_factor, epsilon)
policy_gsvi, values_gsvi, iterations_gsvi = gsvi(tpm, trm, d_factor, epsilon)
policy_rvi, values_rvi, iterations_rvi = rvi(tpm, trm, d_factor, epsilon)

print(f"VI Policy: {policy_vi}, Iterations: {iterations_vi}")
print(f"GSVI Policy: {policy_gsvi}, Iterations: {iterations_gsvi}")
print(f"RVI Policy: {policy_rvi}, Iterations: {iterations_rvi}")
```

## File Structure

- `constants.py`: Global constants (NS, NA, SMALL)
- `vi.py`: Standard Value Iteration Algorithm implementation
- `gsvi.py`: Gauss-Seidel Value Iteration Algorithm implementation
- `rvi.py`: Relative Value Iteration Algorithm implementation
- `main.py`: Main execution module with example MDP
- `__init__.py`: Package initialization
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Algorithm Details

### Standard Value Iteration (VI)

- **Update Rule**: Uses old values for all states
- **Convergence**: Linear convergence with contraction factor $\gamma$
- **Memory**: Requires storing both old and new value functions

### Gauss-Seidel Value Iteration (GSVI)

- **Update Rule**: Uses current values for already-updated states
- **Convergence**: Faster convergence than standard VI
- **Memory**: Updates values in-place
- **Advantage**: More efficient for large state spaces

### Relative Value Iteration (RVI)

- **Update Rule**: Subtracts reference value (state 0) to keep values bounded
- **Convergence**: Similar to standard VI but with bounded values
- **Use Case**: Useful when absolute values are not important

## Algorithm Comparison

| Algorithm | Convergence Speed | Memory Usage | Value Bounds | Use Case |
|-----------|------------------|--------------|--------------|----------|
| **VI** | Standard | High | Unbounded | General purpose |
| **GSVI** | Faster | Low | Unbounded | Large state spaces |
| **RVI** | Standard | High | Bounded | When bounds matter |

## Dependencies

- `numpy`: For numerical computations and array operations

## License

This is a translation of the original C code. Please refer to the original code's license terms. 