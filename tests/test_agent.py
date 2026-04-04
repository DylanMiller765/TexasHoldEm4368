import torch
from torch.distributions import Categorical

from src.policy_network.network import PolicyNet
from src.policy_network.agent import PokerAgent
from src.state_encoder.encoder import encode_state
from src.game_engine.game import Game, Action


# -------------------------
# Helper Policies
# -------------------------
def random_policy(state):
    return Action.CHECK, 0


def agent_policy_factory(agent):
    def policy(state):
        return agent.select_action(state)
    return policy

# -------------------------
# Tests
# -------------------------
def test_model_output():
    print("\n--- test_model_output ---")

    model = PolicyNet()
    game = Game()

    game.play_hand(random_policy)
    state = game.get_state_for_player(0)

    state_vec = encode_state(state)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

    probs = model(state_tensor).squeeze(0)

    print("Probs:", probs)
    print("Sum:", probs.sum().item())

    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()


def test_sampling():
    print("\n--- test_sampling ---")

    model = PolicyNet()
    game = Game()

    game.play_hand(random_policy)
    state = game.get_state_for_player(0)

    state_vec = encode_state(state)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

    probs = model(state_tensor).squeeze(0)
    dist = Categorical(probs)

    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    print("Action:", action.item())
    print("Log prob:", log_prob.item())
    print("Entropy:", entropy.item())

    assert not torch.isnan(log_prob)
    assert not torch.isinf(log_prob)


def test_action_distribution():
    print("\n--- test_action_distribution ---")

    model = PolicyNet()
    game = Game()

    game.play_hand(random_policy)
    state = game.get_state_for_player(0)

    state_vec = encode_state(state)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

    probs = model(state_tensor).squeeze(0)
    dist = Categorical(probs)

    counts = [0] * probs.shape[0]

    for _ in range(100):
        action = dist.sample().item()
        counts[action] += 1

    print("Action counts:", counts)


def test_multiple_states():
    print("\n--- test_multiple_states ---")

    model = PolicyNet()

    for _ in range(5):
        game = Game()
        game.play_hand(random_policy)

        state = game.get_state_for_player(0)
        state_vec = encode_state(state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

        probs = model(state_tensor).squeeze(0)
        print(probs)


def test_full_game():
    print("\n--- test_full_game ---")

    model = PolicyNet()
    agent = PokerAgent(model, encode_state)

    game = Game()
    result = game.play_hand(agent_policy_factory(agent))

    print("Result:", result)


def test_agent_vs_random():
    print("\n--- test_agent_vs_random ---")

    model = PolicyNet()
    agent = PokerAgent(model, encode_state)

    wins = 0
    games = 50

    for _ in range(games):
        game = Game()

        def mixed_policy(state):
            # Player 0 = agent, Player 1 = random
            if state.player_id == 0:
                return agent.select_action(state)
            else:
                return random_policy(state)

        result = game.play_hand(mixed_policy)

        if 0 in result["winners"]:
            wins += 1

    print(f"Win rate: {wins / games:.2f}")

def test_saved_model():
    print("\n--- test_saved_model ---")

    model = PolicyNet()
    agent = PokerAgent(model, encode_state)

    try:
        agent.load("models/poker_model_latest.pt")
    except Exception as e:
        print("Could not load saved model:", e)
        return

    game = Game()
    game.play_hand(random_policy)

    state = game.get_state_for_player(0)
    state_vec = encode_state(state)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

    probs = agent.model(state_tensor).squeeze(0)

    print("Loaded model probs:", probs)
    print("Sum:", probs.sum().item())

    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    test_model_output()
    test_sampling()
    test_action_distribution()
    test_multiple_states()
    test_full_game()
    test_agent_vs_random()
    test_saved_model()