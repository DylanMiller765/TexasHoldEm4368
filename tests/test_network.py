import torch
from torch.distributions import Categorical

from src.policy_network.network import PolicyNet
from src.policy_network.agent import PokerAgent
from src.state_encoder.encoder import encode_state
from src.game_engine.game import Game, Action


# -------------------------
# Helper policy for testing
# -------------------------
def random_policy(state):
    """
    Simple deterministic policy for testing purposes.
    Always checks.
    """
    return Action.CHECK, 0


def model_policy(agent):
    """
    Wraps the agent so it can be used by Game.play_hand()
    """
    def policy(state):
        action, amount = agent.select_action(state)
        return action, amount
    return policy


# -------------------------
# Tests
# -------------------------
def test_model_output():
    print("\n--- Testing model output ---")

    model = PolicyNet()

    game = Game()

    # Run a single hand to ensure proper state initialization
    game.play_hand(random_policy)

    # Now state is guaranteed valid
    state = game.get_state_for_player(0)
    state_vec = encode_state(state)

    print("State vector shape:", state_vec.shape)

    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

    probs = model(state_tensor).squeeze(0)

    print("Probabilities:", probs)
    print("Sum:", probs.sum().item())

    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()


def test_sampling():
    print("\n--- Testing sampling ---")

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

    print("Sampled action:", action.item())
    print("Log prob:", log_prob.item())
    print("Entropy:", entropy.item())

    assert not torch.isnan(log_prob)
    assert not torch.isinf(log_prob)


def test_action_distribution():
    print("\n--- Testing action distribution ---")

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


def test_full_game():
    print("\n--- Testing full game ---")

    model = PolicyNet()
    agent = PokerAgent(model, encode_state)

    game = Game()

    policy = model_policy(agent)

    result = game.play_hand(policy)

    print("Game result:", result)


def test_multiple_states():
    print("\n--- Testing multiple states ---")

    model = PolicyNet()

    for _ in range(5):
        game = Game()
        game.play_hand(random_policy)

        state = game.get_state_for_player(0)
        state_vec = encode_state(state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

        probs = model(state_tensor).squeeze(0)
        print(probs)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    test_model_output()
    test_sampling()
    test_action_distribution()
    test_multiple_states()
    test_full_game()