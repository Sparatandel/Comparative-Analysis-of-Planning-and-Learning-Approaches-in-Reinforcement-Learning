# Comparative Analysis of Planning and Learning Approaches
## in Reinforcement Learning Using Grid World Environments
### — Dark Knight Edition —

---

## Project Overview

This project implements and compares three fundamental Reinforcement Learning algorithms in a GridWorld environment:

| Algorithm | Type | Approach |
|-----------|------|----------|
| **Value Iteration** | Model-Based Planning | Dynamic Programming |
| **Q-Learning** | Model-Free Off-Policy TD | Temporal Difference |
| **SARSA** | Model-Free On-Policy TD | Temporal Difference |

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

---

## Project Structure

```
rl_project/
│
├── main.py          # Main application (Dark Knight UI + Game)
├── algorithms.py    # RL Algorithms: GridWorld, VI, Q-Learning, SARSA
├── analysis.py      # Visualization & Comparative Analysis plots
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## How to Use

### Home Page
- Overview of all three algorithms
- Quick access to Play and Analysis

### PLAY Page
1. **Select Algorithm** — Click on Value Iteration, Q-Learning, or SARSA
2. **Configure Settings** — Grid size and number of training episodes
3. **Train & Play** — Trains the selected algorithm, then launches the game
4. **Train All** — Trains all three for comparative analysis

### Game Page
- Use **WASD** or **Arrow Keys** to move the agent
- **Gold arrow** in each cell shows the AI's recommended action
- **★** marks the goal cell
- **AI Auto-Play** lets the trained agent solve it automatically
- **Reset** to restart the game

### Analysis Page
Tabs available after training all algorithms:
- **Learning Curves** — Reward/steps over episodes, convergence delta
- **Value Maps** — State value heatmaps for each algorithm
- **Policy Grids** — Optimal policy (arrows) for each algorithm
- **Radar Chart** — Multi-dimensional performance comparison
- **Stats Table** — Numerical comparison of all metrics

---

## Algorithm Details

### Value Iteration (Planning)
- Uses Bellman Optimality Equation: `V(s) = max_a Σ P(s'|s,a)[R + γV(s')]`
- Requires full MDP model (transition probabilities + rewards)
- Sweeps all states iteratively until convergence (Δ < θ)
- Extracts greedy policy from converged value function
- **Guaranteed optimal** policy for finite MDPs

### Q-Learning (Off-Policy TD)
- Update: `Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)]`
- Learns from raw experience (no model needed)
- ε-greedy exploration with decay
- Off-policy: target uses greedy max, not actual action taken
- Converges to Q* with appropriate α, ε schedules

### SARSA (On-Policy TD)
- Update: `Q(s,a) += α[r + γQ(s',a') - Q(s,a)]`
- Uses actual next action a' (from ε-greedy policy)
- More conservative — learns value of the exploration policy
- Better in cliff-walking / risk-sensitive environments
- Converges to optimal for GLIE policies

---

## Key Research Questions

1. How does planning (VI) compare to learning (Q/SARSA) in sample efficiency?
2. Does on-policy (SARSA) vs off-policy (Q-Learning) affect path safety?
3. How does grid complexity affect convergence rates?
4. What are the computational tradeoffs between model-based and model-free?

---

## Dependencies
- Python 3.8+
- numpy
- matplotlib
- Pillow (PIL)
- tkinter (standard library)
