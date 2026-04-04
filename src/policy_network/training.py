import os
import torch
import torch.optim as optim
import random

from ..game_engine.game import Game, Action
from ..policy_network.network import PolicyNet
from ..policy_network.agent import PokerAgent
from ..state_encoder.encoder import encode_state


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train(num_episodes=10000):

    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    agent = PokerAgent(model, encode_state)

    negs = 0
    pos = 0

    rewards = []

    for episode in range(num_episodes):

        game = Game()
        agent.reset()

        def get_action(state):
            return agent.select_action(state)

        initial_stack = game.get_state_for_player(0).player_chips

        action = get_action
        
        result = game.play_hand(action)

        final_state = game.get_state_for_player(0)
        final_stack = final_state.player_chips
        pot = final_state.pot_total

        # reward = result["payouts"].get(0, 0)
        # print("raw: ", final_stack - initial_stack)
        reward = final_stack - initial_stack
        reward /= (final_state.current_bet + final_state.opponent_round_bet + 1e-8)
        if abs(reward) > 100000:
            reward = 0.0
        if reward == 0:
            reward = -1
        # print(f"{reward:.3f}")

        loss = agent.compute_loss(reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 1000 == 0:
            path = f"{MODEL_DIR}/poker_model_episode_{episode}.pt"
            agent.save(path, optimizer = optimizer, episode = episode)
            print(f"Saved model to {path}")
            print(agent.model(torch.tensor(agent.encoder(game.get_state_for_player(0)), dtype=torch.float32)))
            model.eval()

            total_profit = 0
            hands = 10000

            for _ in range(hands):
                initial_stack = game.get_state_for_player(0).player_chips

                # run one full hand
                game.play_hand(get_action)
                result = game.get_state_for_player(0)

                final_stack = result.player_chips

                total_profit += (final_stack - initial_stack)

            print("Total Profit:", total_profit)
            print("Avg Profit per Hand:", total_profit / hands)

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward:.4f}")
            print(f"Loss: {loss.item():3f}")
            pass

        if reward > 0:
            pos += 1
        elif reward < 0:
            negs += 1

    print(agent.model(torch.tensor(agent.encoder(game.get_state_for_player(0)), dtype=torch.float32)))

    torch.save(model.state_dict(), f"{MODEL_DIR}/poker_model_latest.pt")


if __name__ == "__main__":
    train()