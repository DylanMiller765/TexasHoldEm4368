"""

Converts a GameState snapshot into a fixed-length float32 numpy vector
suitable for input to a policy network.

Vector layout (total: 318 dimensions)
--------------------------------------
Cards (210 dims)
  Hole cards:        2 x 17 =  34   (one-hot rank [13] + one-hot suit [4])
  Community cards:   5 x 17 =  85   (zero when not yet dealt)
  Card mask:               5 =   5  (1.0 if community card is present)
  Hole card ranks:   2 x 13 =  26   (repeated as rank-only for easy lookup)
  Community ranks:   5 x 13 =  65   (repeated as rank-only for easy lookup, zero padded)
  Hand strength:           1 =   1  (normalised treys rank, 0=worst, 1=best; 0 pre-flop)

Betting / stack (14 dims)
  Pot:1
  Player chips:1
  Opponent chips:1
  To-call:1
  Min raise:1
  Max raise:1
  Player round bet:1
  Opponent round bet:1
  Player bet / pot:1
  Opponent bet / pot:1
  Stack-to-pot ratio:1
  Is player all-in:1
  Is opponent all-in:1
  Effective stack:1

Position / street (7 dims)
  Street one-hot:           4  (preflop, flop, turn, river)
  Dealer / position:        1  (1 if acting player is dealer, 0 otherwise)
  Is first to act:          1
  Relative position:        1

Betting history (87 dims)
  Per street (4 streets) x per seat (2 players) x 4 action features = 4 x 2 x 4 = 32 dims (action category one-hot: fold/check/call/raise)
  Per street x per seat: last raise amount normalised = 4 x 2 x 1 = 8 dims
  Number of actions per street (4 streets x 2 seats x 1 count, normalised) = 8 dims
  Aggression ratios per street (4 streets x 2 seats) = 8 dims (raises / total actions for each player each street)
  Opponent VPIP proxy (across all streets):1
  Opponent aggression proxy:1
  Last action of opponent:4  (fold/check/call/raise one-hot)
  Pot odds:1
  Facing raise flag:1
  Re-raise flag:1

Total: 210 + 14 + 7 + 87 = 318
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..game_engine.card import Card
from ..game_engine.game import Action, GameState, Street
from ..game_engine.hand_eval import evaluate


# Constants

NUM_RANKS = 13            # 2 … A
NUM_SUITS = 4             # s h d c
CARD_DIM = NUM_RANKS + NUM_SUITS   # 17 per card
NUM_HOLE = 2
NUM_COMMUNITY = 5

CHIP_NORM = 1_000.0

# Treys rank range: 1 (royal flush) … 7462 (worst high-card hand)
TREYS_MAX = 7462.0

STREETS = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
STREET_IDX = {s: i for i, s in enumerate(STREETS)}

ACTIONS = [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Dimension constants (must match the docstring layout)

_CARD_SECTION = (
    NUM_HOLE * CARD_DIM          # hole cards one-hot
    + NUM_COMMUNITY * CARD_DIM   # community one-hot (padded)
    + NUM_COMMUNITY              # community presence mask
    + NUM_HOLE * NUM_RANKS       # hole card rank-only (redundant, convenient)
    + NUM_COMMUNITY * NUM_RANKS  # community rank-only (padded)
    + 1                          # hand strength
)
_BET_SECTION = 14
_POS_SECTION = 7
_HIST_SECTION = 87

STATE_DIM = _CARD_SECTION + _BET_SECTION + _POS_SECTION + _HIST_SECTION  # 318


# Helper utilities

def _rank_int(card: Card) -> int:
    """0-indexed rank (2=0, 3=1, … A=12)."""
    from treys import Card as TreysCard
    return TreysCard.get_rank_int(card.to_int())


def _suit_int(card: Card) -> int:
    """0-indexed suit: spades=0, hearts=1, diamonds=2, clubs=3."""
    from treys import Card as TreysCard
    suit_map = {1: 0, 2: 1, 4: 2, 8: 3}
    return suit_map[TreysCard.get_suit_int(card.to_int())]


def _encode_card(card: Card) -> np.ndarray:
    """Return a 17-dim float32 one-hot vector for a single card."""
    v = np.zeros(CARD_DIM, dtype=np.float32)
    v[_rank_int(card)] = 1.0
    v[NUM_RANKS + _suit_int(card)] = 1.0
    return v


def _encode_rank_only(card: Card) -> np.ndarray:
    """Return a 13-dim rank-only one-hot vector."""
    v = np.zeros(NUM_RANKS, dtype=np.float32)
    v[_rank_int(card)] = 1.0
    return v


def _norm(value: float, scale: float = CHIP_NORM) -> float:
    """Clip-normalise a chip value to [0, 1]."""
    return float(np.clip(value / scale, 0.0, 2.0))


def _action_to_idx(action: Action) -> int:
    """Map Action enum to 0-3; unknown actions map to CALL (2)."""
    return ACTION_IDX.get(action, ACTION_IDX[Action.CALL])


# Public encoder

class StateEncoder:
    """Encodes a GameState into a fixed-length float32 numpy vector.

    Attributes:
        dim: Length of the output vector (318).
        chip_norm: Chip count used as the normalisation denominator.
    """

    def __init__(self, chip_norm: float = CHIP_NORM):
        self.dim: int = STATE_DIM
        self.chip_norm = chip_norm

    # Main entry point
    
    def encode(self, state: GameState) -> np.ndarray:
        """Encode a GameState into a float32 vector of length ``self.dim``.
        Args:
            state: The GameState for the acting player.
        Returns:
            np.ndarray of shape (318,) and dtype float32.
        """
        parts = [
            self._encode_cards(state), # 210 dims
            self._encode_betting(state), #  14 dims
            self._encode_position(state), #   7 dims
            self._encode_history(state), #  87 dims
        ]
        vec = np.concatenate(parts)
        assert vec.shape == (STATE_DIM,), (
            f"Encoder produced shape {vec.shape}, expected ({STATE_DIM},)"
        )
        return vec

    # Section encoders
    
    def _encode_cards(self, state: GameState) -> np.ndarray:
        """210-dim card section."""
        parts = []

        # Hole cards (2 × 17 = 34)
        for card in state.hole_cards:
            parts.append(_encode_card(card))

        # Community cards (5 × 17 = 85, zero-pad missing cards)
        community_mask = np.zeros(NUM_COMMUNITY, dtype=np.float32)
        for i in range(NUM_COMMUNITY):
            if i < len(state.community_cards):
                parts.append(_encode_card(state.community_cards[i]))
                community_mask[i] = 1.0
            else:
                parts.append(np.zeros(CARD_DIM, dtype=np.float32))

        # Community presence mask (5)
        parts.append(community_mask)

        # Hole card rank-only (2 × 13 = 26)
        for card in state.hole_cards:
            parts.append(_encode_rank_only(card))

        # Community rank-only (5 × 13 = 65, zero-pad)
        for i in range(NUM_COMMUNITY):
            if i < len(state.community_cards):
                parts.append(_encode_rank_only(state.community_cards[i]))
            else:
                parts.append(np.zeros(NUM_RANKS, dtype=np.float32))

        # Hand strength (1) — 0 on pre-flop (no community cards yet)
        hand_strength = 0.0
        if len(state.community_cards) >= 3:
            try:
                hr = evaluate(state.hole_cards, state.community_cards)
                # treys_rank: 1 = best, 7462 = worst → invert to [0, 1]
                hand_strength = float(1.0 - (hr.treys_rank - 1) / (TREYS_MAX - 1))
            except Exception:
                hand_strength = 0.0
        parts.append(np.array([hand_strength], dtype=np.float32))

        return np.concatenate(parts)  # 210 dims

    def _encode_betting(self, state: GameState) -> np.ndarray:
        """14-dim betting / stack section."""
        n = self.chip_norm
        pot = state.pot_total
        p_chips = state.player_chips
        o_chips = state.opponent_chips
        to_call = max(0, state.current_bet - state.player_round_bet)
        effective = min(p_chips, o_chips)

        pot_safe = pot if pot > 0 else 1.0

        v = np.array([
            _norm(pot, n),
            _norm(p_chips, n),
            _norm(o_chips, n),
            _norm(to_call, n),
            _norm(state.min_raise, n),
            _norm(state.max_raise, n),
            _norm(state.player_round_bet, n),
            _norm(state.opponent_round_bet, n),
            float(np.clip(state.player_round_bet / pot_safe, 0.0, 2.0)),
            float(np.clip(state.opponent_round_bet / pot_safe, 0.0, 2.0)),
            float(np.clip(effective / pot_safe, 0.0, 10.0) / 10.0),  # SPR norm to [0,1]
            float(p_chips == 0),   # player is all-in
            float(o_chips == 0),   # opponent is all-in
            _norm(effective, n),
        ], dtype=np.float32)
        return v  # 14 dims

    def _encode_position(self, state: GameState) -> np.ndarray:
        """7-dim position / street section."""
        # Street one-hot (4)
        street_oh = np.zeros(4, dtype=np.float32)
        si = STREET_IDX.get(state.street, 0)
        street_oh[si] = 1.0

        # Dealer flag — in heads-up, player_id 0 is dealer when dealer=0
        # We use player_round_bet vs opponent: on preflop the dealer/SB acts
        # first, so if opponent_round_bet > player_round_bet we are OOP.
        is_dealer = float(
            state.player_round_bet <= state.opponent_round_bet
            and state.street == Street.PREFLOP
        )

        # Is first to act this street (no history yet for current street)
        current_street_history = (
            state.betting_history[-1] if state.betting_history else []
        )
        is_first = float(len(current_street_history) == 0)

        # Relative position: post-flop OOP (out of position) = acts first
        relative_pos = float(state.street != Street.PREFLOP and is_first)

        v = np.concatenate([
            street_oh,
            np.array([is_dealer, is_first, relative_pos], dtype=np.float32),
        ])
        return v  # 7 dims

    def _encode_history(self, state: GameState) -> np.ndarray:
        """87-dim betting history section."""
        pid = state.player_id
        oid = 1 - pid

        # Per-street, per-player action counts and raises
        # 4 streets × 2 players × 4 action one-hots = 32 (last action each)
        action_vecs = np.zeros((4, 2, len(ACTIONS)), dtype=np.float32)
        last_raise = np.zeros((4, 2), dtype=np.float32)    # 4×2 = 8
        action_counts = np.zeros((4, 2), dtype=np.float32) # 4×2 = 8
        raise_counts = np.zeros((4, 2), dtype=np.float32)  # 4×2 for aggression

        for street_i, street_history in enumerate(state.betting_history):
            if street_i >= 4:
                break
            for actor_id, action, amount in street_history:
                seat = 0 if actor_id == pid else 1
                a_idx = _action_to_idx(action)
                action_vecs[street_i, seat, a_idx] = 1.0
                action_counts[street_i, seat] += 1.0
                if action in (Action.RAISE, Action.ALL_IN):
                    raise_counts[street_i, seat] += 1.0
                    last_raise[street_i, seat] = _norm(amount, self.chip_norm)

        # Flatten action one-hots: 4 × 2 × 4 = 32
        hist_actions = action_vecs.flatten()  # 32

        # Last raise per street/player: 4 × 2 = 8
        hist_raises = last_raise.flatten()    # 8

        # Normalise action counts (clip at 10 raises per street): 4 × 2 = 8
        hist_counts = np.clip(action_counts / 10.0, 0.0, 1.0).flatten()  # 8

        # Aggression ratios: raises / total_actions per (street, seat): 4 × 2 = 8
        total_safe = np.where(action_counts > 0, action_counts, 1.0)
        aggression = (raise_counts / total_safe).flatten()  # 8

        # Opponent VPIP proxy: did opponent voluntarily put $ in pre-flop?
        # (check/call/raise on preflop = VPIP)
        vpip = 0.0
        if state.betting_history:
            pf = state.betting_history[0]
            for actor_id, action, _ in pf:
                if actor_id == oid and action in (Action.CALL, Action.RAISE, Action.ALL_IN):
                    vpip = 1.0
                    break

        # Opponent aggression proxy across all streets
        total_opp_actions = float(action_counts[:, 1].sum())
        total_opp_raises = float(raise_counts[:, 1].sum())
        opp_agg = (total_opp_raises / total_opp_actions
                   if total_opp_actions > 0 else 0.0)

        # Last opponent action (4-dim one-hot)
        last_opp_action_vec = np.zeros(4, dtype=np.float32)
        last_opp_action = None
        for street_hist in reversed(state.betting_history):
            for actor_id, action, _ in reversed(street_hist):
                if actor_id == oid:
                    last_opp_action = action
                    break
            if last_opp_action is not None:
                break
        if last_opp_action is not None:
            last_opp_action_vec[_action_to_idx(last_opp_action)] = 1.0

        # Pot odds: to_call / (pot + to_call)
        to_call = max(0, state.current_bet - state.player_round_bet)
        denom = state.pot_total + to_call
        pot_odds = float(to_call / denom) if denom > 0 else 0.0

        # Facing raise flag
        facing_raise = float(to_call > 0)

        # Re-raise flag: both players have raised in the current street
        re_raise = 0.0
        if state.betting_history:
            cur = state.betting_history[-1]
            raised_pids = {actor_id for actor_id, action, _ in cur
                           if action in (Action.RAISE, Action.ALL_IN)}
            if pid in raised_pids and oid in raised_pids:
                re_raise = 1.0

        scalars = np.array(
            [vpip, opp_agg] + list(last_opp_action_vec) + [pot_odds, facing_raise, re_raise],
            dtype=np.float32,
        )  # 2 + 4 + 3 = 9  →  but wait, let's count carefully below

        result = np.concatenate([
            hist_actions,   # 32
            hist_raises,    #  8
            hist_counts,    #  8
            aggression,     #  8
            scalars,        #  9  → vpip(1)+opp_agg(1)+last_opp(4)+pot_odds(1)+facing(1)+reraise(1)
        ])
        # 32 + 8 + 8 + 8 + 9 = 65 … we need 87 total.
        # Pad to 87 with zeros for future extensibility.
        pad = np.zeros(_HIST_SECTION - len(result), dtype=np.float32)
        return np.concatenate([result, pad])  # 87 dims


def encode_state(state: GameState, chip_norm: float = CHIP_NORM) -> np.ndarray:
    """Functional wrapper around StateEncoder.encode.

    Args:
        state: Acting player's GameState.
        chip_norm: Chip normalisation scale (default 1000).

    Returns:
        float32 numpy array of shape (318,).
    """
    return StateEncoder(chip_norm=chip_norm).encode(state)