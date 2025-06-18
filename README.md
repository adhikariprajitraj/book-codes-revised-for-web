# Optimization Algorithms for Decision Making and Learning

This repository contains implementations of fundamental algorithms for optimization, decision making, and machine learning. The codebase is organized into four main categories: **Markov Decision Processes (MDPs)**, **Neural Networks**, **Reinforcement Learning (RL)**, and **Semi-Markov Decision Processes (SMDPs)**.

## Table of Contents

1. [Overview](#overview)
2. [Markov Decision Processes (MDPs)](#markov-decision-processes-mdps)
3. [Neural Networks](#neural-networks)
4. [Reinforcement Learning (RL)](#reinforcement-learning-rl)
5. [Semi-Markov Decision Processes (SMDPs)](#semi-markov-decision-processes-smdps)
6. [Installation and Usage](#installation-and-usage)
7. [Mathematical Foundations](#mathematical-foundations)

## Overview

This repository provides implementations of key algorithms in optimization and decision theory:

- **MDPs**: Policy Iteration and Value Iteration for average reward problems
- **Neural Networks**: Backpropagation with batch and online learning
- **Reinforcement Learning**: Q-Learning for discounted and average reward problems
- **SMDPs**: Policy Iteration for semi-Markov decision processes

Each algorithm is implemented in C with Python translations available for some components.

## Markov Decision Processes (MDPs)

### Mathematical Framework

A **Markov Decision Process (MDP)** is defined by the tuple $(S, A, P, R, \gamma)$ where:

- $S$: Set of states
- $A$: Set of actions
- $P: S \times A \times S \rightarrow [0,1]$: Transition probability function
- $R: S \times A \times S \rightarrow \mathbb{R}$: Reward function
- $\gamma \in [0,1]$: Discount factor (for discounted problems)

### Average Reward Criterion

For average reward MDPs, the objective is to maximize the long-term average reward:

<p align="center">
  $$\rho^{*} = \max_{\pi} \lim_{T \rightarrow \infty} \frac{1}{T} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T-1} R(s_t, a_t, s_{t+1}) \right]$$
</p>



where $\pi: S \rightarrow A$ is a policy.

### Policy Iteration Algorithm

**Policy Iteration** alternates between policy evaluation and policy improvement:

#### Policy Evaluation
For a given policy $\pi$, solve the system of equations:

$$V^\pi(s) = \sum_{s'} P(s'|s, \pi(s)) \left[ R(s, \pi(s), s') + V^\pi(s') \right] - \rho^\pi$$

where $\rho^\pi$ is the average reward of policy $\pi$.

#### Policy Improvement
Update the policy greedily:

$$\pi'(s) = \arg\max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + V^\pi(s') \right]$$

### Value Iteration Algorithm

**Value Iteration** directly updates the value function:

$$V_{k+1}(s) = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + V_k(s') \right]$$

#### Relative Value Iteration (RVI)
To handle unbounded values in average reward problems, RVI subtracts a reference value:

$$V_{k+1}(s) = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + V_k(s') \right] - V_k(s_0)$$

where $s_0$ is a reference state.

### Implementation Details

- **Directory**: `mdp/`
- **Policy Iteration**: `mdp/avg/pol/` (C) and `mdp_avg_pol_python/` (Python)
- **Value Iteration**: `mdp/avg/val/` (C) and `mdp_avg_val_python/` (Python)

## Neural Networks

### Mathematical Framework

A **feedforward neural network** with one hidden layer computes:

$$y = f_2(W_2 \cdot f_1(W_1 \cdot x + b_1) + b_2)$$

where:
- $x \in \mathbb{R}^{n_{in}}$: Input vector
- $W_1 \in \mathbb{R}^{n_{hidden} \times n_{in}}$: Input-to-hidden weights
- $W_2 \in \mathbb{R}^{n_{out} \times n_{hidden}}$: Hidden-to-output weights
- $b_1, b_2$: Bias vectors
- $f_1, f_2$: Activation functions (typically sigmoid)

### Sigmoid Activation Function

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

with derivative:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### Backpropagation Algorithm

**Backpropagation** computes gradients using the chain rule:

#### Forward Pass
$$z_1 = W_1 \cdot x + b_1$$
$$h = f_1(z_1)$$
$$z_2 = W_2 \cdot h + b_2$$
$$y = f_2(z_2)$$

#### Backward Pass
$$\delta_2 = (y - t) \odot f_2'(z_2)$$
$$\delta_1 = (W_2^T \cdot \delta_2) \odot f_1'(z_1)$$

#### Weight Updates
$$\Delta W_2 = \delta_2 \cdot h^T$$
$$\Delta W_1 = \delta_1 \cdot x^T$$
$$\Delta b_2 = \delta_2$$
$$\Delta b_1 = \delta_1$$

### Implementation Details

- **Directory**: `neural-nets/`
- **Batch Learning**: `neural-nets/batch-newrnd/`
- **Online Learning**: `neural-nets/neuron-newrnd/`

## Reinforcement Learning (RL)

### Mathematical Framework

**Reinforcement Learning** learns optimal policies through interaction with an environment. The Q-function represents the expected future reward:

$$Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) | s_0 = s, a_0 = a \right]$$

### Q-Learning Algorithm

**Q-Learning** is an off-policy temporal difference learning algorithm:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R(s_t, a_t, s_{t+1}) + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

where:
- $\alpha$: Learning rate
- $\gamma$: Discount factor

### Average Reward Q-Learning

For average reward problems, the update rule becomes:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R(s_t, a_t, s_{t+1}) - \rho + \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

where $\rho$ is the estimated average reward.

### Implementation Details

- **Directory**: `rl/`
- **Discrete Q-Learning**: `rl/disc-newrnd/`
- **Average Reward Q-Learning**: `rl/avg-newrnd/mdp-newrnd/`
- **SMDP Q-Learning**: `rl/avg-newrnd/smdp-newrnd/`

## Semi-Markov Decision Processes (SMDPs)

### Mathematical Framework

A **Semi-Markov Decision Process (SMDP)** extends MDPs by allowing actions to take variable time. It's defined by the tuple $(S, A, P, R, F)$ where:

- $S$: Set of states
- $A$: Set of actions
- $P: S \times A \times S \rightarrow [0,1]$: Transition probability function
- $R: S \times A \times S \rightarrow \mathbb{R}$: Reward function
- $F: S \times A \times S \times \mathbb{R}^+ \rightarrow [0,1]$: Holding time distribution

### SMDP Policy Iteration

For average reward SMDPs, the policy evaluation equation becomes:

$$V^\pi(s) = \sum_{s'} P(s'|s, \pi(s)) \left[ R(s, \pi(s), s') + V^\pi(s') \right] - \rho^\pi \tau(s, \pi(s))$$

where $\tau(s, a)$ is the expected holding time for action $a$ in state $s$.

### Implementation Details

- **Directory**: `smdp/`
- **Algorithm**: Policy Iteration for average reward SMDPs

## Installation and Usage

### Prerequisites

- **C Compiler**: GCC or Clang
- **Python**: Python 3.7+ (for Python implementations)
- **NumPy**: For Python implementations

### Compiling C Code

```bash
# Compile MDP algorithms
cd mdp/avg/pol
gcc -o main main.c pia.c solver.c -lm

# Compile neural network
cd neural-nets/batch-newrnd
gcc -o main main.c backprop_batch.c compute_sigmoid.c decoder.c encoder.c evaluator.c generator.c init_net.c max_int.c reader.c simu_net.c unifrnd.c -lm

# Compile Q-Learning
cd rl/disc-newrnd
gcc -o main main.c qlearn.c action_selector.c initialize.c jump_learn.c pol_finder.c simulator_mc.c state_finder.c unifrnd.c -lm

# Compile SMDP
cd smdp
gcc -o main main.c pias.c solver.c -lm
```

### Running Python Implementations

```bash
# Install dependencies
pip install numpy

# Run MDP Policy Iteration
cd mdp_avg_pol_python
python3 main.py

# Run MDP Value Iteration
cd mdp_avg_val_python
python3 main.py
```

## Mathematical Foundations

### Convergence Properties

#### Policy Iteration
- **Convergence**: Guaranteed to converge to optimal policy in finite steps
- **Complexity**: $O(|S|^3)$ per iteration for policy evaluation
- **Rate**: Linear convergence

#### Value Iteration
- **Convergence**: Guaranteed to converge to optimal value function
- **Complexity**: $O(|S|^2|A|)$ per iteration
- **Rate**: Linear convergence with contraction factor $\gamma$

#### Q-Learning
- **Convergence**: Converges to optimal Q-function with probability 1
- **Requirements**: All state-action pairs visited infinitely often
- **Rate**: Depends on learning rate schedule

### Optimality Conditions

#### Bellman Optimality Equation
For discounted MDPs:
$$V^*(s) = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

For average reward MDPs:
$$V^*(s) + \rho^* = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + V^*(s') \right]$$

#### Q-Function Optimality
$$Q^*(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]$$

### Error Bounds

#### Value Iteration Error Bound
$$\|V_k - V^*\|_\infty \leq \frac{\gamma^k}{1-\gamma} \|V_0 - V^*\|_\infty$$

#### Q-Learning Error Bound
With appropriate learning rate schedule:
$$\|Q_k - Q^*\|_\infty \leq \epsilon \text{ with high probability}$$

## Applications

### MDPs
- **Resource allocation**: Optimal resource distribution
- **Inventory management**: Optimal ordering policies
- **Robot navigation**: Path planning in uncertain environments

### Neural Networks
- **Pattern recognition**: Classification and regression
- **Function approximation**: Universal function approximation
- **Feature learning**: Automatic feature extraction

### Reinforcement Learning
- **Game playing**: AlphaGo, game AI
- **Autonomous systems**: Self-driving cars, robotics
- **Recommendation systems**: Personalized content delivery

### SMDPs
- **Maintenance scheduling**: Optimal maintenance policies
- **Queue management**: Service rate optimization
- **Manufacturing**: Production scheduling

## Contributing

This repository contains educational implementations of fundamental algorithms. Contributions are welcome for:

- Bug fixes and improvements
- Additional algorithms
- Better documentation
- Performance optimizations
- Python translations of remaining C code

## License

This code is provided for educational and research purposes. Please refer to individual file headers for specific licensing information.

## References

1. Puterman, M. L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
3. Haykin, S. (2009). *Neural Networks and Learning Machines*
4. Howard, R. A. (1960). *Dynamic Programming and Markov Processes* 
