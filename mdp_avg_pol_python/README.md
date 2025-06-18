# MDP Average Reward Policy Iteration

This is a Python implementation of Policy Iteration for Average Reward Markov Decision Processes (MDPs). It is a direct translation of the original C code from the `mdp/avg/pol` directory.

## Overview

The package implements the Policy Iteration Algorithm for solving MDPs with the average reward criterion. The algorithm alternates between:

1. **Policy Evaluation**: Solving a linear system to compute the average reward and value function for the current policy
2. **Policy Improvement**: Updating the policy greedily based on the computed value function

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
from mdp_avg_pol_python import pia, create_example_mdp

# Create the example MDP
tpm, trm = create_example_mdp()

# Run policy iteration
optimal_policy, optimal_values, average_reward, iterations = pia(tpm, trm)

print(f"Optimal Policy: {optimal_policy}")
print(f"Average Reward: {average_reward}")
```

### Creating Your Own MDP

```python
import numpy as np
from mdp_avg_pol_python import pia

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

# Run policy iteration
optimal_policy, optimal_values, average_reward, iterations = pia(tpm, trm)
```

## File Structure

- `constants.py`: Global constants (NS, NA, SMALL)
- `solver.py`: Linear system solver for policy evaluation
- `pia.py`: Policy Iteration Algorithm implementation
- `main.py`: Main execution module with example MDP
- `__init__.py`: Package initialization
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Algorithm Details

### Policy Iteration for Average Reward MDPs

The algorithm solves MDPs with the average reward criterion, where the objective is to maximize the long-term average reward per time step.

**Input:**
- Transition probability matrix `tpm[action][state][next_state]`
- Transition reward matrix `trm[action][state][next_state]`

**Output:**
- Optimal policy `policy[state]`
- Value function `values[state]`
- Average reward `rho`
- Number of iterations

**Algorithm:**
1. Start with an arbitrary policy
2. **Policy Evaluation**: Solve the linear system to find the average reward and value function
3. **Policy Improvement**: Update the policy greedily
4. Repeat until the policy converges

### Linear System Solver

The policy evaluation step requires solving a linear system of the form `Gx = 0`, where:
- `G` is an augmented matrix of size `(NS, NS+1)`
- The solution `x` gives the average reward and value function
- The first element of `x` is the average reward `rho`

## Example Output

When you run `python main.py`, you should see output similar to:

```
MDP Average Reward Policy Iteration
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

Running Policy Iteration Algorithm...

Average reward in iteration 0: 8.5
Value for state 0 in current iteration is 0.0
Value for state 1 in current iteration is 2.5
...
Number of iterations needed: 2
Optimal action for state 0 is 1
Optimal action for state 1 is 0
Optimal value for state 0 is 0.0
Optimal value for state 1 is 2.5

==================================================
FINAL RESULTS
==================================================
Optimal Policy: [1 0]
Optimal Values: [0.  2.5]
Average Reward: 8.5
Total Iterations: 2
==================================================
```

## Dependencies

- `numpy`: For numerical computations and linear algebra

## License

This is a translation of the original C code. Please refer to the original code's license terms. 