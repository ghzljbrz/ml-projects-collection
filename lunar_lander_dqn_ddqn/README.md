# 🚀 Lunar Lander: DQN & Double DQN Hyperparameter Study

**A deep reinforcement learning project that trains an agent to safely land a spacecraft in OpenAI Gym's LunarLander-v3 environment, with a systematic sweep over training episodes and batch size, and a head-to-head comparison of DQN vs. Double DQN.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DQN%20%2F%20DDQN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-LunarLander--v3-0081A5?style=for-the-badge">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

**LunarLander-v3** is a continuous-state, discrete-action control problem: an agent must fire its engines to land a spacecraft gently between two flags, balancing fuel efficiency against a safe, low-velocity touchdown. It's a standard benchmark for deep reinforcement learning precisely because naive approaches tend to either crash, hover indefinitely, or fail to generalize across episodes.

This project trains a **Deep Q-Network (DQN)** agent on LunarLander-v3 and runs a genuine hyperparameter study — sweeping **5 episode budgets × 3 batch sizes (15 configurations total)** — before implementing **Double DQN (DDQN)** to test whether decoupling action-selection from action-evaluation reduces the Q-value overestimation bias that vanilla DQN is known for.

The core question:

> How do training duration and batch size jointly affect DQN's landing performance on LunarLander-v3, and does Double DQN meaningfully outperform standard DQN under the same conditions?

## ✨ What This Project Demonstrates

- 🧠 **Deep Q-Network implementation in PyTorch** — a 3-hidden-layer (512 units each) network with `LayerNorm` and `Dropout` for training stability, paired with a **soft-updated target network** and an **experience replay buffer** (25,000-transition capacity).
- 🎛️ **Genuine hyperparameter search, not guesswork** — training 15 separate DQN configurations across episode counts (50, 100, 150, 200, 250) and batch sizes (32, 64, 128), then comparing final average reward across the full grid.
- ⚖️ **DQN vs. Double DQN comparison** — implementing DDQN's core fix (using the online network to *select* the best action and the target network to *evaluate* it) and training it under matching conditions to isolate the architectural difference's real effect.
- 🎥 **Reproducible, visual evaluation** — recording landing videos at regular intervals via `RecordVideo`, alongside quantitative reward curves for every configuration, so performance claims are both measurable and visually verifiable.
- 🖥️ **Practical deep-RL engineering** — GPU-aware training (`torch.device`), Colab virtual-display setup for headless video rendering, and checkpointing across long-running experiments.

## 🎮 Environment

- **Task:** [`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (Gymnasium) — land a spacecraft between two flags using four discrete actions (do nothing, fire left/main/right engine).
- **Reward signal:** shaped reward based on proximity to the landing pad, velocity, angle, leg contact, and fuel usage, with large bonuses/penalties for successful landing or crashing.

## 🤖 Notebook

### DQN & DDQN Hyperparameter Study — `lunar_lander_dqn_ddqn.ipynb`
Implements and trains:
- **DQN agent:** the core network, experience replay buffer, and training loop, swept across a full 5×3 grid of episode counts and batch sizes (15 runs).
- **DDQN agent:** the same architecture with the Double DQN target-computation fix, trained at 100 and 250 episodes (batch size 128) for direct comparison against the equivalent DQN runs.
- Reward-curve plots and recorded landing videos for every configuration.

## 📈 Results Snapshot

**DQN — final average reward by episode budget and batch size:**

| Episodes | Batch 32 | Batch 64 | Batch 128 |
|---|---|---|---|
| 50 | −93.6 | −183.4 | −136.8 |
| 100 | 162.5 | −187.0 | −62.2 |
| 150 | −15.6 | −40.3 | 176.8 |
| 200 | **247.0** (best overall) | 76.2 | 154.8 |
| 250 | 185.1 | 8.8 | 135.9 |

> A commonly used threshold for "solving" LunarLander is an average reward of ~200 over 100 episodes; several configurations (32/200, 128/150, 128/200) crossed or approached that bar.

**DQN vs. DDQN (batch size 128):**

| Episodes | DQN | DDQN |
|---|---|---|
| 100 | −62.2 | −135.6 (at ep. 43, run in progress) |
| 250 | 135.9 | *(see notebook for full curve)* |

> 📌 **Key finding:** performance was **highly non-monotonic with respect to episode count** — more training episodes did not reliably mean better landings (e.g. batch 32 peaked at 200 episodes and *dropped* at 250), underscoring that in deep RL, training longer isn't automatically better without adequate exploration decay and stability tuning. **Batch size 32 with 200 episodes produced the best single result** (average reward 247.0) across the entire 15-run DQN grid, suggesting that for this network size and replay buffer, smaller batches with a moderate training budget generalized better than larger batches or longer runs.

## 🗃️ Project Structure

```
.
├── lunar_lander_dqn_ddqn.ipynb   # DQN hyperparameter sweep (15 configs) + DDQN comparison
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install torch gymnasium[box2d] matplotlib pyvirtualdisplay moviepy
```

> The notebook was developed in Google Colab and uses a virtual display (`pyvirtualdisplay`) for headless rendering and video recording of landing episodes; a GPU runtime is recommended for the full hyperparameter sweep.

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Gymnasium (LunarLander-v3)` · `Matplotlib` · `MoviePy`

## 🔮 Future Improvements

- Complete and directly compare full DQN vs. DDQN reward curves (both at 100 and 250 episodes) rather than a partial run, to draw a firmer conclusion about DDQN's benefit.
- Evaluate each trained agent over multiple held-out test episodes (not just the training curve) to get a variance-aware performance estimate, since single-run training curves are noisy.
- Explore Dueling DQN or Prioritized Experience Replay as further extensions, given the modest and non-monotonic gains observed from batch-size/episode tuning alone.
- Automate the hyperparameter grid with a loop and results table instead of duplicated per-configuration cells, to make the sweep easier to extend and reproduce.

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
