# UAV_Final
## Overview
This project implements and compares multiple optimization and learning-based methods, including heuristic, tabular reinforcement learning, and deep reinforcement learning approaches.


## File Descriptions


- **DQNLearningAgent.py**
This file implements a **Deep Q-Learning (DQN)** agent. It uses a neural network to approximate the Q-function, enabling the agent to handle large or continuous state spaces more effectively than tabular methods.


- **QLearningAgent.py**
This file implements **Tabular Q-Learning**, where Q-values are stored explicitly in a table. This approach is suitable for problems with a relatively small and discrete state-action space.


## Execution Steps


1. **Run `main.py`**
First, execute `main.py` to process and evaluate four different methods:
- FPA (Flower Pollination Algorithm)
- Tabular Q-Learning
- Deep Q-Learning
- Brute Force


2. **Run `parse.py`**
After running `main.py`, execute `parse.py` to extract **SINR values** and save them into `.txt` files.


3. **Run `plot.py`**
Using the generated `.txt` files, run `plot.py` to draw the **CCDF (Complementary Cumulative Distribution Function)** curves for performance comparison.


## Output


- `.txt` files containing extracted SINR values
- CCDF plots generated from the SINR data for different methods


## Notes


- Ensure all dependencies are properly installed before running the scripts.
- The scripts should be executed in the order specified above to obtain correct results.