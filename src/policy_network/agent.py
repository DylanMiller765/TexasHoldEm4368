import torch
import numpy as np
from torch.distributions import Categorical
from ..game_engine.game import Action
import random


class PokerAgent:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.log_probs = []
        self.baseline = 0.0
        self.rewards = []

    def reset(self):
        """Call at start of each hand"""
        self.log_probs = []

    def select_action(self, state):
        state_vec = self.encoder(state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)

        probs = self.model(state_tensor)
        dist = Categorical(probs)

        self.entropy = dist.entropy()

        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        self.log_probs.append(log_prob)
        print(probs)
        if action_idx.item() == 0:
            return Action.FOLD, 0

        elif action_idx.item() == 1:
            to_call = state.current_bet - state.player_round_bet
            if to_call <= 0:
                return Action.CHECK, 0
            else:
                return Action.CALL, 0

        else:
            percs = probs.tolist()
            rating = percs[2]
            ag_raise = state.min_raise
            while True:
                chance = random.random()
                if chance <= rating:
                    ag_raise += int((state.min_raise)*chance)
                else:
                    break

            return Action.RAISE, ag_raise

    def compute_loss(self, reward):
        self.rewards.append(reward)
        mean = np.mean(self.rewards)
        std = np.std(self.rewards)
        self.baseline = 0.99 * self.baseline + 0.1 * reward
        loss = 0
        advantage = (reward - self.baseline - mean)/(std + 1e-8)
        for log_prob in self.log_probs:
            loss += -log_prob * advantage - 0.03 * self.entropy
        return loss

    def save(self, path, optimizer=None, episode=None):
        checkpoint = {
            "model_state": self.model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()

        if episode is not None:
            checkpoint["episode"] = episode

        torch.save(checkpoint, path)

    def load(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        self.model.load_state_dict(checkpoint)

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_episode = checkpoint.get("episode", 0)
        return start_episode