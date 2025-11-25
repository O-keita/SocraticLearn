# SocraticLearn: Reinforcement Learning Socratic Tutor

**Student:** Omar Keita  
**Repository:** [https://github.com/O-keita/SocraticLearn.git](https://github.com/O-keita/SocraticLearn.git)

## Project Overview

SocraticLearn is an intelligent tutoring system powered by reinforcement learning that adapts teaching strategies in real-time to maximize student learning outcomes. The system uses a custom Gymnasium environment to simulate dynamic student states (engagement, confusion, effort) and trains RL agents to select optimal pedagogical interventions through trial and error.

This project compares four state-of-the-art RL algorithms—DQN, REINFORCE, A2C, and PPO—to determine which approach best models adaptive, personalized education.

## Environment Description

### Agent
An AI Socratic tutor that selects instructional actions to guide students through learning sessions, optimizing for engagement, reduced confusion, and increased effort.

### Action Space (Discrete - 5 Actions)

| Action ID | Description | Effect on Student |
|-----------|-------------|-------------------|
| 0 | Ask Socratic question | ↑ Engagement, ↑ Confusion, ↑ Effort |
| 1 | Give hint | ↑ Engagement, ↓ Confusion, ↑ Effort |
| 2 | Provide code example | ↓ Engagement, ↓ Confusion, ↓ Effort |
| 3 | Encourage reflection | ↑↑ Engagement, ↓ Confusion, ↑↑ Effort |
| 4 | Ask student to explain | ↑ Engagement, ↓ Confusion, ↑ Effort |

**Note:** Actual effects are modulated by hidden student skill, stochastic noise, and diminishing returns for action repetition.

### Observation Space
- **Engagement level** (0–1)
- **Confusion level** (0–1)
- **Effort level** (0–1)
- **Current performance score**
- **Steps taken in session**

### Reward Structure
Dynamic reward system based on **actual learning improvement**:

**Delta-Based Rewards:**
- `+50 × Δ(engagement)` - Rewards increased student engagement
- `+40 × Δ(confusion reduction)` - Rewards decreased confusion
- `+20 × state progression` - Bonus for advancing learning states (e.g., Confused → Engaged → Mastery)

**Penalties:**
- `-5.0` if confusion > 0.9 (student overwhelmed)
- `-8.0` if engagement < 0.05 (student disengaged)
- `-0.1` per step (time penalty for teaching efficiency)

**Key Features:**
- Rewards reflect measured pedagogical impact, not fixed action values
- Stochastic noise adds environmental uncertainty
- Diminishing returns for repeating the same action
- Hidden per-episode student skill parameter varies responsiveness

## Project Structure

```
SocraticLearn/
├── assets/                    # Student and teacher sprites
├── demos/                     # Visualization demos
│   ├── dqn_visualization.py
│   └── random_demo.py
├── environment/               # Custom Gymnasium environment
│   ├── custom_env.py
│   └── rendering.py
├── evaluate/                  # Model evaluation scripts
│   ├── dqn_evaluate.py
│   ├── a2c_evaluate.py
│   ├── ppo_evaluate.py
│   └── reinforce_evaluate.py
├── training/                  # Training scripts
│   ├── dqn_training.py
│   ├── a2c_training.py
│   ├── ppo_training.py
│   └── reinforce_training.py
├── models/                    # Saved trained models
│   ├── dqn_engagement/
│   ├── a2c_engagement/
│   ├── ppo_engagement/
│   └── reinforce_engagement/
├── tensorboard/               # Training logs
├── main.py                    # Main execution script
├── requirements.txt           # Dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/O-keita/SocraticLearn.git
cd SocraticLearn
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Best Model
Execute the main script to run the best-performing trained agent:
```bash
python main.py
```

### Training Models

Train individual algorithms:
```bash
# DQN
python training/dqn_training.py

# REINFORCE
python training/reinforce_training.py

# A2C
python training/a2c_training.py

# PPO
python training/ppo_training.py
```

### Evaluating Models
```bash
python evaluate/dqn_evaluate.py
python evaluate/reinforce_evaluate.py
python evaluate/a2c_evaluate.py
python evaluate/ppo_evaluate.py
```

### Visualization Demos
```bash
# Random agent demo
python demos/random_demo.py

# DQN trained agent visualization
python demos/dqn_visualization.py
```

### Monitoring Training (TensorBoard)
```bash
tensorboard --logdir=./tensorboard
```
Open http://localhost:6006 in your browser.

## Algorithms Implemented

### 1. Deep Q-Network (DQN)
**Value-based method** using experience replay and target networks.

**Best Hyperparameters:**
- Learning Rate: 0.0001
- Gamma: 0.99
- Replay Buffer: 100,000
- Batch Size: 32
- Epsilon: 0.1 → 0.01
- **Mean Reward: 337.00**

### 2. REINFORCE
**Policy gradient method** using Monte Carlo returns.

**Best Hyperparameters:**
- Learning Rate: 0.003
- Gamma: 0.99
- Episodes: 3000
- **Mean Reward: 345.8**

### 3. Advantage Actor-Critic (A2C)
**Actor-critic method** with parallel environments.

**Best Hyperparameters:**
- Learning Rate: 0.0007
- Gamma: 0.99
- Entropy Coefficient: 0.01
- n_steps: 5
- **Mean Reward: 133.59**

### 4. Proximal Policy Optimization (PPO)
**Advanced policy gradient** with clipped surrogate objective.

**Best Hyperparameters:**
- Learning Rate: 0.0003
- n_steps: 50
- Clip Range: 0.2
- Entropy Coefficient: 0.01
- **Mean Reward: 314.43**

## Key Results

| Algorithm | Mean Reward | Training Stability | Convergence Speed |
|-----------|-------------|-------------------|-------------------|
| **REINFORCE** | **345.8** | Moderate | Slow |
| **DQN** | **337.00** | High | Medium |
| **PPO** | **314.43** | Very High | Fast |
| **A2C** | **133.59** | Moderate | Fast |

### Key Findings
- **REINFORCE** achieved the highest mean reward but with higher variance
- **DQN** showed excellent stability with competitive performance
- **PPO** demonstrated the best balance between performance and training stability
- **A2C** converged quickly but plateaued at lower rewards

## Hyperparameter Tuning

The project includes extensive hyperparameter exploration:
- **DQN**: 10+ configurations tested
- **REINFORCE**: 8+ configurations tested
- **A2C**: 11+ configurations tested  
- **PPO**: 10+ configurations tested

See the report for complete hyperparameter tables and analysis.

## Dependencies

Key libraries:
- `gymnasium` - Environment framework
- `stable-baselines3` - RL algorithms
- `torch` - Neural networks
- `pygame` - Visualization
- `tensorboard` - Training monitoring
- `numpy`, `matplotlib` - Data processing and visualization

See `requirements.txt` for complete list.

## Video Demonstration

**[Video Recording Link]** - 3-minute demonstration showing:
- Problem statement
- Agent behavior and reward structure
- Best model performance with GUI and terminal outputs
- Performance analysis

## Contributing

This is an academic project for reinforcement learning coursework. For questions or suggestions, please open an issue in the repository.

## License

This project is for educational purposes as part of a university assignment.

## Acknowledgments

- Built using Stable-Baselines3 library
- Custom Gymnasium environment implementation
- Inspired by adaptive learning systems and Socratic teaching methods

---

**For detailed results, analysis, and visualizations, refer to the project report.**