# Texas Hold'em Poker AI

An AI system that plays Texas Hold'em Poker in a two-player format, trained via self-play reinforcement learning.

## Project Overview

The project will develop a system that plays Texas Hold'em Poker in a two player format. The system will be composed of four components: a game engine, a state encoder, a policy network, and an evaluation module.

**Game Engine** — Simulates the full Texas Hold'em rule set. Handles deck management, card dealing, betting rounds (pre-flop, flop, turn, and river), pot calculations, and hand evaluation.

**State Encoder** — Converts the raw game state into a vector. Inputs include the agent's hole cards, the community cards, the current pot size, stack sizes, position, and the opponent's betting history. Cards are encoded using a combination of rank and suit representations.

**Policy Network** — Takes the state vector as input and outputs a probability distribution over fold, call, or raise. The network is trained using a Reinforcement Learning loop in which the agent plays repeated hands against a copy of itself, updating network weights based on chip gain and loss as the reward signal. Self-play allows the agent to improve without requiring a fixed opponent. Opponent hand range estimation is maintained throughout each hand using updates conditioned on observed betting actions. At each decision point, the agent computes value estimates across possible opponent holdings to inform the policy output.

**Evaluation Module** — Benchmarks the trained agent against a rule-based baseline opponent, recording win rate and chip differential as the performance metrics.

## Project Structure

```
PokerBot/
├── src/
│   ├── game_engine/       # Texas Hold'em game logic
│   │   ├── __init__.py
│   │   ├── card.py        # Card and Deck classes
│   │   ├── hand_eval.py   # Hand ranking and evaluation
│   │   ├── game.py        # Game loop and betting rounds
│   │   └── pot.py         # Pot and side-pot management
│   ├── state_encoder/     # Game state to vector conversion
│   │   ├── __init__.py
│   │   └── encoder.py     # State encoding logic
│   ├── policy_network/    # RL agent and neural network
│   │   ├── __init__.py
│   │   ├── network.py     # Policy network architecture
│   │   ├── agent.py       # Agent decision-making
│   │   └── training.py    # Self-play training loop
│   └── evaluation/        # Benchmarking and metrics
│       ├── __init__.py
│       ├── baseline.py    # Rule-based baseline opponent
│       └── evaluate.py    # Win rate and chip differential tracking
├── tests/
│   ├── __init__.py
│   ├── test_game_engine.py
│   ├── test_encoder.py
│   ├── test_network.py
│   └── test_evaluation.py
├── models/                # Saved model checkpoints
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train the agent via self-play
python -m src.policy_network.training

# Evaluate against baseline
python -m src.evaluation.evaluate

# Runs the UI and allows the player to play against the AI
python server.py

#Open http://localhost:5000/ and play
```
