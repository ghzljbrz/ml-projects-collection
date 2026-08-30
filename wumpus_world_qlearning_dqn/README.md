# 🏹 Wumpus World: Q-Learning vs. Deep Q-Network (DQN)

**A reinforcement learning project that trains an autonomous agent to navigate the classic Wumpus World — hunting for gold while avoiding pits and a deadly Wumpus — comparing tabular Q-Learning against a Deep Q-Network.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Keras%20DQN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Q--Learning-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

The **Wumpus World** is a classic AI benchmark environment: an agent explores a 4×4 grid containing hidden pits, a dangerous Wumpus, and a pot of gold, using only local percepts to decide whether to move, shoot an arrow, or grab treasure. It's a compact but genuinely hard reinforcement-learning problem — sparse rewards, irreversible failure states (falling in a pit, encountering the Wumpus), and a partially-hazardous action space (a wasted arrow shot can't be undone).

This project implements and trains **two fundamentally different RL agents** to solve it:

- A **tabular Q-Learning agent**, learning a direct state-action value table through trial and error.
- A **Deep Q-Network (DQN) agent**, using a neural network to approximate Q-values, backed by experience replay and a target network for stable learning.

The core question:

> Does a neural-network-based DQN agent learn a more effective policy for Wumpus World than a classical tabular Q-Learning agent — and how does that show up in survival rate, cumulative reward, and reward stability over training?

## ✨ What This Project Demonstrates

- 🎮 **Custom RL environment design** — a from-scratch `GridEnvironment` implementing the full Wumpus World rules: agent movement, percepts, arrow-shooting mechanics, pit/Wumpus death conditions, and a custom reward structure (+1000 for gold, −1000 for death, +500 for killing the Wumpus, −10 per move).
- 📊 **Tabular Q-Learning from scratch** — implementing the Q-learning update rule with an ε-greedy exploration strategy and exploration-rate decay, trained over thousands of episodes.
- 🧠 **Deep Q-Network implementation** — a Keras-based DQN with a **target network** (for stable Q-value targets) and a **replay memory buffer** (for decorrelated, i.i.d.-like training batches) — the two key stabilization techniques that make deep RL tractable.
- 📈 **Rigorous training diagnostics** — tracking total reward, cumulative reward, and mean reward per episode for both agents, visualized side-by-side to compare learning dynamics, not just final performance.
- ☠️ **Safety-aware evaluation** — explicitly tracking death counts across training, treating survivability as a first-class metric alongside reward, which matters in any RL problem with irreversible failure states.
- ⚖️ **Sample-efficiency comparison** — evaluating both agents' performance across different training budgets (1,000 vs. 5,000 episodes for Q-Learning) to understand how each method scales with experience.

## 🎯 Environment Design

- **Grid:** 4×4, agent starts at `(0,0)`.
- **Gold:** at `(3,3)` — the goal state.
- **Pits:** at `(1,1)` and `(2,2)` — instant death on entry.
- **Wumpus:** at `(1,3)` — can be avoided, encountered (death), or shot with a single available arrow.
- **Actions:** movement (up/down/left/right) plus shooting the arrow in each direction (8 actions total for the DQN agent).
- **Rewards:** `+1000` reaching the gold, `−1000` falling in a pit or meeting a live Wumpus, `+500` for successfully shooting the Wumpus, `−10` per move (encouraging efficient paths).

## 🤖 Notebook

### Wumpus World RL Agent — `wumpus_world_qlearning_dqn.ipynb`
Implements both agents end-to-end on a shared Wumpus World environment:
- **Q-Learning agent:** tabular state-action values updated via the Bellman equation, trained for 1,000 and 5,000 episodes to study the effect of training budget.
- **DQN agent:** a two-hidden-layer (128 units each) neural network approximating Q-values, trained with experience replay and a periodically-updated target network over 1,000 episodes.
- Reward-curve visualizations (total / cumulative / mean reward per episode) for both agents.

## 📈 Results Snapshot

| Metric | Q-Learning (1,000 ep.) | Q-Learning (5,000 ep.) | DQN (1,000 ep.) |
|---|---|---|---|
| Total reward (summed) | −148,613 | **+155,372** | −60,100 |
| Highest single-episode reward | 145 | 145 | 145 |
| Lowest single-episode reward | −1,173 | −1,212 | −1,100 |
| Deaths | 219 | 288 | 172 |
| Final-episode accuracy | — | — | **82.8%** |

> 📌 **Key finding:** tabular Q-Learning needed **5x more episodes** (5,000 vs. 1,000) to flip from a large net-negative total reward to a strongly positive one, while the **DQN agent achieved a much smaller net loss and fewer deaths within just 1,000 episodes**, and reached 82.8% accuracy by its final episode — suggesting the learned function approximation generalizes across similar states faster than a purely tabular approach, at the cost of significantly higher per-episode computational overhead (neural network forward/backward passes vs. simple table lookups).

## 🗃️ Project Structure

```
.
├── wumpus_world_qlearning_dqn.ipynb   # Custom Wumpus World env + Q-Learning and DQN agents
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install numpy matplotlib tensorflow
```

> The notebook was developed in Google Colab; the environment and both agents are self-contained with no external dataset download required.

## 🛠️ Tech Stack

`Python` · `NumPy` · `TensorFlow / Keras` · `Matplotlib`

## 🔮 Future Improvements

- Run Q-Learning and DQN for an equal number of episodes to make the total-reward comparison strictly apples-to-apples.
- Add a moving-average survival-rate plot to visualize how death frequency evolves during training for each agent.
- Experiment with reward shaping (e.g. small percept-based bonuses) to see if it accelerates early learning for either agent.
- Extend to a Double DQN or Dueling DQN architecture to test whether they further reduce the DQN agent's death rate.
- Evaluate both trained agents on a fixed test suite of random start positions, rather than only reporting training-time statistics.

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
