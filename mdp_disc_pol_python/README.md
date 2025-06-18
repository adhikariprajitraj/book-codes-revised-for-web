# MDP Discounted Policy Iteration

This is a Python implementation of Policy Iteration for Discounted Markov Decision Processes (MDPs). It is a direct translation of the original C code from the `mdp/disc/pol` directory.

## Overview

The package implements Policy Iteration for solving MDPs with the discounted reward criterion. The algorithm alternates between policy evaluation and policy improvement until convergence to find the optimal policy.

## Mathematical Framework

### Discounted MDP Objective

For discounted MDPs, the objective is to maximize the discounted sum of future rewards:

$$\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) \right]$$

where $\gamma \in [0,1)$ is the discount factor.

### Policy Iteration Algorithm

**Policy Iteration** alternates between policy evaluation and policy improvement:

#### Policy Evaluation
For a given policy $\pi$, solve the system of equations:

$$V^\pi(s) = \sum_{s'} P(s'|s, \pi(s)) \left[ R(s, \pi(s), s') + \gamma V^\pi(s') \right]$$

#### Policy Improvement
Update the policy greedily:

$$\pi'(s) = \arg\max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

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
from mdp_disc_pol_python import pid, create_example_mdp

# Create the example MDP
tpm, trm = create_example_mdp()

# Set discount factor
d_factor = 0.8

# Run policy iteration
optimal_policy, optimal_values, iterations = pid(tpm, trm, d_factor)

print(f"Optimal Policy: {optimal_policy}")
print(f"Optimal Values: {optimal_values}")
```

### Creating Your Own MDP

```python
import numpy as np
from mdp_disc_pol_python import pid

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
d_factor = 0.8
policy, values, iterations = pid(tpm, trm, d_factor)
```

## File Structure

- `constants.py`: Global constants (NS, NA, SMALL)
- `solver.py`: Linear system solver for policy evaluation
- `pid.py`: Policy Iteration Algorithm implementation
- `main.py`: Main execution module with example MDP
- `__init__.py`: Package initialization
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Algorithm Details

### Policy Iteration for Discounted MDPs

The algorithm solves MDPs with the discounted reward criterion, where future rewards are worth less than immediate rewards.

**Input:**
- Transition probability matrix `tpm[action][state][next_state]`
- Transition reward matrix `trm[action][state][next_state]`
- Discount factor `d_factor` (gamma)

**Output:**
- Optimal policy `policy[state]`
- Value function `values[state]`
- Number of iterations

**Algorithm:**
1. Start with an arbitrary policy
2. **Policy Evaluation**: Solve the linear system to find the value function for the current policy
3. **Policy Improvement**: Update the policy greedily
4. Repeat until the policy converges

### Linear System Solver

The policy evaluation step requires solving a linear system of the form $Gx = b$, where:
- $G$ is a matrix of size $(NS, NS)$
- $x$ is the value function vector
- $b$ is the reward vector

## Example Output

When you run `python3 main.py`, you should see output similar to:

```
MDP Discounted Policy Iteration
Python Implementation

==================================================
MDP INFORMATION
==================================================
Number of states: 2
Number of actions: 2
Discount factor: 0.8

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

Running Discounted Policy Iteration Algorithm...
Discount factor (gamma): 0.8

Value for state 0 in current iteration is 53.03333333333333
Value for state 1 in current iteration is 51.86666666666667
Number of iterations needed: 2
Optimal action for state 0 is 1
Optimal action for state 1 is 0
Optimal value for state 0 is 53.03333333333333
Optimal value for state 1 is 51.86666666666667

==================================================
FINAL RESULTS
==================================================
Optimal Policy: [1 0]
Optimal Values: [53.03333333 51.86666667]
Total Iterations: 2
==================================================
```

## Key Differences from Average Reward MDPs

| Aspect | Discounted MDPs | Average Reward MDPs |
|--------|-----------------|---------------------|
| **Objective** | Maximize discounted sum | Maximize long-term average |
| **Value Function** | Finite, bounded | May grow unbounded |
| **Convergence** | Linear convergence | Linear convergence |
| **Use Case** | Finite horizon effect | Steady-state behavior |

## Dependencies

- `numpy`: For numerical computations and linear algebra

## License

This is a translation of the original C code. Please refer to the original code's license terms. 