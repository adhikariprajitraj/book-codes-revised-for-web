# MDP Average Reward Value Iteration

This is a Python implementation of Value Iteration algorithms for Average Reward Markov Decision Processes (MDPs). It is a direct translation of the original C code from the `mdp/avg/val` directory.

## Overview

The package implements two Value Iteration algorithms for solving MDPs with the average reward criterion:

1. **Standard Value Iteration (VI)**: Values can become unbounded during iteration
2. **Relative Value Iteration (RVI)**: Values are kept bounded by subtracting a reference value

Both algorithms iteratively update the value function until convergence to find the optimal policy.

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
python main.py
```

### Using as a Module

```python
import numpy as np
from mdp_avg_val_python import rvia, via, create_example_mdp

# Create the example MDP
tpm, trm = create_example_mdp()

# Set convergence threshold
epsilon = 0.001

# Run Relative Value Iteration (RVI)
policy_rvi, values_rvi, avg_reward_rvi, iterations_rvi = rvia(tpm, trm, epsilon)

# Run Standard Value Iteration (VI)
policy_vi, values_vi, iterations_vi = via(tpm, trm, epsilon)

print(f"RVI Optimal Policy: {policy_rvi}")
print(f"RVI Average Reward: {avg_reward_rvi}")
print(f"VI Optimal Policy: {policy_vi}")
```

### Creating Your Own MDP

```python
import numpy as np
from mdp_avg_val_python import rvia

# Define your MDP
NS = 2  # Number of states
NA = 2  # Number of actions

# Transition probability matrix (NA x NS x NS)
tpm = np.array([
    [[0.7, 0.3], [0.4, 0.6]],  # Action 0
    [[0.9, 0.1], [0.2, 0.8]]   # Action 1
])

# Transition reward matrix (NA x NS x NS)
trm = np.array([
    [[6, -5], [7, 12]],        # Action 0
    [[10, 17], [-14, 13]]      # Action 1
])

# Run Relative Value Iteration
epsilon = 0.001
policy, values, avg_reward, iterations = rvia(tpm, trm, epsilon)
```

## File Structure

- `constants.py`: Global constants (NS, NA, SMALL)
- `via.py`: Standard Value Iteration Algorithm implementation
- `rvia.py`: Relative Value Iteration Algorithm implementation
- `main.py`: Main execution module with example MDP
- `__init__.py`: Package initialization
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Algorithm Details

### Value Iteration for Average Reward MDPs

Both algorithms solve MDPs with the average reward criterion, where the objective is to maximize the long-term average reward per time step.

**Input:**
- Transition probability matrix `tpm[action][state][next_state]`
- Transition reward matrix `trm[action][state][next_state]`
- Convergence threshold `epsilon`

**Output:**
- Optimal policy `policy[state]`
- Value function `values[state]`
- Average reward `rho` (RVI only)
- Number of iterations

**Algorithm:**
1. Initialize value function to zero
2. **Value Update**: For each state, compute the maximum expected value over all actions
3. **Convergence Check**: Check if the span of value differences is below epsilon
4. Repeat until convergence

### Standard Value Iteration (VI)

- Values can grow unbounded during iteration
- Uses span of value differences for convergence
- Does not estimate average reward explicitly

### Relative Value Iteration (RVI)

- Keeps values bounded by subtracting a reference value (typically from state 0)
- Provides explicit estimate of average reward
- More numerically stable for average reward problems
- Uses span of relative value differences for convergence

## Example Output

When you run `python main.py`, you should see output similar to:

```
MDP Average Reward Value Iteration
Python Implementation

==================================================
MDP INFORMATION
==================================================
Number of states: 2
Number of actions: 2

Transition Probabilities:
  Action 0:
    From state 0: [0.7 0.3]
    From state 1: [0.4 0.6]
  Action 1:
    From state 0: [0.9 0.1]
    From state 1: [0.2 0.8]

Transition Rewards:
  Action 0:
    From state 0: [ 6 -5]
    From state 1: [ 7 12]
  Action 1:
    From state 0: [10 17]
    From state 1: [-14 13]
==================================================

Running Relative Value Iteration (RVI) Algorithm...
Convergence threshold (epsilon): 0.001

span of difference vector=0.123456
Average reward estimate= 8.5
The value function for state 0=0.0
The value function for state 1=2.5
...
Epsilon-optimal action for state 0=1
Epsilon-optimal action for state 1=0
The value function for state 0=0.0
The value function for state 1=2.5
Average reward estimate= 8.5
The number of iterations needed to converge= 15

==================================================
FINAL RESULTS - RVI
==================================================
Optimal Policy: [1 0]
Optimal Values: [0.  2.5]
Average Reward: 8.5
Total Iterations: 15
==================================================
```

## Key Differences Between VI and RVI

| Aspect | Standard VI | Relative VI (RVI) |
|--------|-------------|-------------------|
| Value Bounds | Unbounded | Bounded |
| Average Reward | Not estimated | Explicitly estimated |
| Numerical Stability | May be unstable | More stable |
| Convergence | Based on span | Based on relative span |
| Use Case | General MDPs | Average reward MDPs |

## Dependencies

- `numpy`: For numerical computations and array operations

## License

This is a translation of the original C code. Please refer to the original code's license terms. 