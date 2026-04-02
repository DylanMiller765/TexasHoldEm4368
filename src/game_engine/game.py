"""Game loop and betting rounds for Texas Hold'em.

This module provides the core Game class that orchestrates a full hand of
No-Limit Texas Hold'em between two players.  Players interact through a
simple callback interface (the ``player_action`` callable), making the engine
agnostic to whether the player is a human, a rule-based bot, or an RL agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .card import Card, Deck
from .hand_eval import evaluate, compare_hands
from .pot import Pot


class Action(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()
    ALL_IN = auto()


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


@dataclass
class Player:
    player_id: int
    chips: int
    hole_cards: list[Card] = field(default_factory=list)
    is_folded: bool = False
    is_all_in: bool = False
    total_bet_this_hand: int = 0

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.is_folded = False
        self.is_all_in = False
        self.total_bet_this_hand = 0


@dataclass
class GameState:
    """A snapshot of the current game state, passed to players each decision.

    This is the public information a player (or state encoder) needs to make
    a decision.  Hole cards are only for the acting player.
    """
    street: Street
    community_cards: list[Card]
    pot_total: int
    current_bet: int           # the bet amount the player must match
    player_chips: int          # acting player's remaining chips
    opponent_chips: int        # opponent's remaining chips
    hole_cards: list[Card]     # acting player's private cards
    min_raise: int             # minimum legal raise-to amount
    max_raise: int             # maximum legal raise-to amount (= player chips)
    player_round_bet: int      # how much acting player already bet this round
    opponent_round_bet: int    # how much opponent already bet this round
    player_id: int
    betting_history: list[list[tuple[int, Action, int]]]  # per-street history


class Game:
    """Manages a single hand of heads-up No-Limit Texas Hold'em.

    Args:
        stacks: Starting chip counts for [player_0, player_1].
        small_blind: Small blind amount (default 10).
        big_blind: Big blind amount (default 20).
        deck_seed: Optional RNG seed for reproducible deals.
    """

    def __init__(
        self,
        stacks: tuple[int, int] = (1000, 1000),
        small_blind: int = 10,
        big_blind: int = 20,
        deck_seed: Optional[int] = None,
    ):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck(seed=deck_seed)
        self.pot = Pot()
        self.community_cards: list[Card] = []
        self.street = Street.PREFLOP
        self.dealer: int = 0  # index of the dealer/button (0 or 1)
        self.players = [
            Player(player_id=0, chips=stacks[0]),
            Player(player_id=1, chips=stacks[1]),
        ]
        self.betting_history: list[list[tuple[int, Action, int]]] = []
        self.hand_over = False
        self.winner_ids: list[int] = []
        self.payouts: dict[int, int] = {}

    # ── public API ──────────────────────────────────────────────────────

    def play_hand(self, get_action) -> dict:
        """Play one complete hand.

        Args:
            get_action: Callable(GameState) -> (Action, int).
                Called each time a player must act.  Must return an
                (Action, amount) tuple.  ``amount`` is only used for
                Action.RAISE (the raise-to total); ignored for other actions.

        Returns:
            Dict with hand result:
                - "winners": list of winning player ids
                - "payouts": {player_id: chips_won}
                - "community_cards": final board
                - "hands": {player_id: hole_cards}
                - "hand_ranks": {player_id: HandRank} (if showdown)
        """
        self._start_hand()
        self._post_blinds()

        streets = [
            (Street.PREFLOP, 0),
            (Street.FLOP, 3),
            (Street.TURN, 1),
            (Street.RIVER, 1),
        ]

        for street, n_cards in streets:
            self.street = street
            self.betting_history.append([])

            if street != Street.PREFLOP and not self.hand_over:
                self._deal_community(n_cards)

            if not self.hand_over:
                self._betting_round(get_action)

            if self.hand_over:
                break

        result = self._resolve()
        return result

    def get_state_for_player(self, player_id: int) -> GameState:
        """Build a GameState snapshot for the given player."""
        p = self.players[player_id]
        opp = self.players[1 - player_id]
        current_bet = self.pot.player_round_bet(opp.player_id)
        player_round_bet = self.pot.player_round_bet(player_id)
        to_call = current_bet - player_round_bet

        min_raise = current_bet + self.big_blind
        max_raise = p.chips + player_round_bet  # raise-to at most all chips

        return GameState(
            street=self.street,
            community_cards=list(self.community_cards),
            pot_total=self.pot.total,
            current_bet=current_bet,
            player_chips=p.chips,
            opponent_chips=opp.chips,
            hole_cards=list(p.hole_cards),
            min_raise=min_raise,
            max_raise=max_raise,
            player_round_bet=player_round_bet,
            opponent_round_bet=self.pot.player_round_bet(opp.player_id),
            player_id=player_id,
            betting_history=[list(s) for s in self.betting_history],
        )

    # ── hand lifecycle ──────────────────────────────────────────────────

    def _start_hand(self):
        """Reset state and deal hole cards."""
        self.deck.reset()
        self.pot.reset()
        self.community_cards.clear()
        self.betting_history.clear()
        self.hand_over = False
        self.winner_ids.clear()
        self.payouts.clear()
        for p in self.players:
            p.reset_for_new_hand()

        # Deal 2 hole cards to each player
        for p in self.players:
            p.hole_cards = self.deck.deal(2)

    def _post_blinds(self):
        """Post small and big blinds.  In heads-up the dealer posts the SB."""
        sb_player = self.players[self.dealer]
        bb_player = self.players[1 - self.dealer]

        sb_amount = min(self.small_blind, sb_player.chips)
        bb_amount = min(self.big_blind, bb_player.chips)

        self._bet(sb_player, sb_amount)
        self._bet(bb_player, bb_amount)

    def _deal_community(self, n: int):
        """Deal n community cards (burn one first)."""
        self.deck.deal(1)  # burn card
        self.community_cards.extend(self.deck.deal(n))

    # ── betting ─────────────────────────────────────────────────────────

    def _can_act_count(self) -> int:
        """Number of players who are not folded and not all-in."""
        return sum(
            1 for p in self.players if not p.is_folded and not p.is_all_in
        )

    def _betting_round(self, get_action):
        """Run a single betting round until action is closed."""
        # Skip entirely if fewer than 2 players can act (all-in / folded)
        if self._can_act_count() < 2:
            self.pot.new_round()
            return

        # In heads-up preflop: dealer/SB acts first.
        # Post-flop: non-dealer acts first.
        if self.street == Street.PREFLOP:
            first = self.dealer
        else:
            first = 1 - self.dealer

        order = [first, 1 - first]
        actors = list(order)  # queue of players to act

        i = 0
        while i < len(actors):
            pid = actors[i]
            p = self.players[pid]

            if p.is_folded or p.is_all_in:
                i += 1
                continue

            opp = self.players[1 - pid]
            if opp.is_folded:
                break

            # If opponent is all-in and we've already matched their bet,
            # no further action is needed
            if opp.is_all_in:
                to_call = (self.pot.player_round_bet(opp.player_id)
                           - self.pot.player_round_bet(pid))
                if to_call <= 0:
                    i += 1
                    continue

            state = self.get_state_for_player(pid)
            action, amount = get_action(state)
            action, amount = self._validate_action(pid, action, amount)
            self._apply_action(pid, action, amount)

            self.betting_history[-1].append((pid, action, amount))

            if action == Action.FOLD:
                self.hand_over = True
                self.winner_ids = [1 - pid]
                return

            if action in (Action.RAISE, Action.ALL_IN):
                # Give opponent another chance to act (if they can)
                if not opp.is_all_in and not opp.is_folded:
                    if i + 1 >= len(actors) or actors[i + 1] != (1 - pid):
                        actors.append(1 - pid)

            i += 1

        # End of round — reset round bets
        self.pot.new_round()

    def _validate_action(self, player_id: int, action: Action, amount: int) -> tuple[Action, int]:
        """Validate and normalize a player action."""
        p = self.players[player_id]
        opp = self.players[1 - player_id]
        to_call = self.pot.player_round_bet(opp.player_id) - self.pot.player_round_bet(player_id)
        to_call = max(0, to_call)

        if action == Action.FOLD:
            return (Action.FOLD, 0)

        if action == Action.CHECK:
            if to_call > 0:
                # Can't check when there's a bet — treat as fold
                return (Action.FOLD, 0)
            return (Action.CHECK, 0)

        if action == Action.CALL:
            call_amount = min(to_call, p.chips)
            if call_amount >= p.chips:
                return (Action.ALL_IN, p.chips)
            return (Action.CALL, call_amount)

        if action in (Action.RAISE, Action.ALL_IN):
            # amount = raise-to total for this round
            min_raise_to = self.pot.player_round_bet(opp.player_id) + self.big_blind
            max_raise_to = p.chips + self.pot.player_round_bet(player_id)

            if amount >= max_raise_to or action == Action.ALL_IN:
                # All-in
                return (Action.ALL_IN, p.chips)

            if amount < min_raise_to:
                # Below min raise — just call
                call_amount = min(to_call, p.chips)
                if call_amount >= p.chips:
                    return (Action.ALL_IN, p.chips)
                return (Action.CALL, call_amount)

            actual_cost = amount - self.pot.player_round_bet(player_id)
            actual_cost = min(actual_cost, p.chips)
            return (Action.RAISE, actual_cost)

        return (Action.FOLD, 0)

    def _apply_action(self, player_id: int, action: Action, amount: int):
        """Apply a validated action."""
        p = self.players[player_id]

        if action == Action.FOLD:
            p.is_folded = True
        elif action == Action.CHECK:
            pass
        elif action in (Action.CALL, Action.RAISE):
            self._bet(p, amount)
        elif action == Action.ALL_IN:
            self._bet(p, amount)
            p.is_all_in = True

    def _bet(self, player: Player, amount: int):
        """Transfer chips from player to pot."""
        amount = min(amount, player.chips)
        player.chips -= amount
        player.total_bet_this_hand += amount
        self.pot.add(player.player_id, amount)
        if player.chips == 0:
            player.is_all_in = True

    # ── resolution ──────────────────────────────────────────────────────

    def _resolve(self) -> dict:
        """Determine winner and distribute pot."""
        result = {
            "community_cards": list(self.community_cards),
            "hands": {p.player_id: list(p.hole_cards) for p in self.players},
        }

        if self.hand_over and self.winner_ids:
            # Someone folded
            winners = self.winner_ids
        else:
            # Showdown
            self.street = Street.SHOWDOWN
            hands_to_compare = [p.hole_cards for p in self.players if not p.is_folded]
            player_ids = [p.player_id for p in self.players if not p.is_folded]

            rankings = {
                pid: evaluate(self.players[pid].hole_cards, self.community_cards)
                for pid in player_ids
            }
            best_rank = max(rankings.values())
            winners = [pid for pid, r in rankings.items() if r == best_rank]
            result["hand_ranks"] = rankings

        # Handle side pots when players have put in unequal amounts
        active_players = [p for p in self.players if not p.is_folded]
        contributions = {p.player_id: p.total_bet_this_hand for p in active_players}
        if len(set(contributions.values())) > 1:
            # Unequal contributions — need side pots
            active_ids = set(contributions.keys())
            self.pot.build_side_pots(contributions, active_ids)

        self.payouts = self.pot.settle(winners)
        for pid, chips in self.payouts.items():
            self.players[pid].chips += chips

        self.winner_ids = winners
        result["winners"] = winners
        result["payouts"] = dict(self.payouts)
        return result

    def advance_dealer(self):
        """Move the dealer button for the next hand."""
        self.dealer = 1 - self.dealer
