"""Tests for the Texas Hold'em game engine."""

import pytest
from src.game_engine.card import Card, Deck
from src.game_engine.hand_eval import HandCategory, HandRank, evaluate, compare_hands
from src.game_engine.pot import Pot
from src.game_engine.game import Game, GameState, Action, Street, Player


# ── helpers ─────────────────────────────────────────────────────────────

def _cards(*specs):
    """_cards("As", "Kh") -> [Card("As"), Card("Kh")]"""
    return [Card(s) for s in specs]


# ── Card & Deck ─────────────────────────────────────────────────────────

class TestCard:
    def test_card_creation(self):
        c = Card("As")
        assert c.rank_char == "A"
        assert c.suit_char == "s"

    def test_card_repr(self):
        assert repr(Card("As")) == "As"
        assert repr(Card("Th")) == "Th"

    def test_card_equality(self):
        assert Card("Kd") == Card("Kd")
        assert Card("Kd") != Card("Kh")

    def test_card_hash(self):
        s = {Card("Qc"), Card("Qc")}
        assert len(s) == 1

    def test_card_ordering(self):
        assert Card("2c") < Card("Ac")


class TestDeck:
    def test_deck_has_52_cards(self):
        d = Deck(seed=42)
        assert len(d) == 52

    def test_deal(self):
        d = Deck(seed=42)
        cards = d.deal(5)
        assert len(cards) == 5
        assert len(d) == 47

    def test_reset(self):
        d = Deck(seed=42)
        d.deal(10)
        assert len(d) == 42
        d.reset()
        assert len(d) == 52


# ── Hand Evaluation ─────────────────────────────────────────────────────

class TestHandEvaluation:
    def test_high_card(self):
        hole = _cards("2c", "5d")
        board = _cards("8h", "Js", "Ac", "7d", "3s")
        r = evaluate(hole, board)
        assert r.category == HandCategory.HIGH_CARD

    def test_one_pair(self):
        hole = _cards("7c", "7d")
        board = _cards("2h", "9s", "Ac", "4d", "Js")
        r = evaluate(hole, board)
        assert r.category == HandCategory.ONE_PAIR

    def test_two_pair(self):
        hole = _cards("7c", "9d")
        board = _cards("7d", "9s", "Ac", "2h", "3s")
        r = evaluate(hole, board)
        assert r.category == HandCategory.TWO_PAIR

    def test_three_of_a_kind(self):
        hole = _cards("7c", "7d")
        board = _cards("7h", "9s", "Ac", "2d", "4s")
        r = evaluate(hole, board)
        assert r.category == HandCategory.THREE_OF_A_KIND

    def test_straight(self):
        hole = _cards("5c", "6d")
        board = _cards("7h", "8s", "9c", "2d", "3s")
        r = evaluate(hole, board)
        assert r.category == HandCategory.STRAIGHT

    def test_ace_low_straight(self):
        hole = _cards("Ac", "2d")
        board = _cards("3h", "4s", "5c", "9d", "Js")
        r = evaluate(hole, board)
        assert r.category == HandCategory.STRAIGHT

    def test_flush(self):
        hole = _cards("2h", "5h")
        board = _cards("8h", "Jh", "Ah", "3c", "9d")
        r = evaluate(hole, board)
        assert r.category == HandCategory.FLUSH

    def test_full_house(self):
        hole = _cards("7c", "7d")
        board = _cards("7h", "9s", "9c", "2d", "4s")
        r = evaluate(hole, board)
        assert r.category == HandCategory.FULL_HOUSE

    def test_four_of_a_kind(self):
        hole = _cards("7c", "7d")
        board = _cards("7h", "7s", "Ac", "2d", "4s")
        r = evaluate(hole, board)
        assert r.category == HandCategory.FOUR_OF_A_KIND

    def test_straight_flush(self):
        hole = _cards("5h", "6h")
        board = _cards("7h", "8h", "9h", "2c", "3d")
        r = evaluate(hole, board)
        assert r.category == HandCategory.STRAIGHT_FLUSH

    def test_royal_flush_is_straight_flush(self):
        # Treys classifies royal flush as straight flush
        hole = _cards("Ts", "Js")
        board = _cards("Qs", "Ks", "As", "2c", "3d")
        r = evaluate(hole, board)
        assert r.category == HandCategory.STRAIGHT_FLUSH

    def test_hand_comparison(self):
        board = _cards("8d", "Jc", "3s", "2c", "9d")
        pair = evaluate(_cards("7c", "7d"), board)
        flush_board = _cards("2h", "5d", "8c", "Jh", "9h")
        flush = evaluate(_cards("3h", "7h"), flush_board)
        assert flush > pair

    def test_pair_tiebreak(self):
        board = _cards("2h", "5s", "9c", "3d", "4s")
        pair_kings = evaluate(_cards("Kc", "Kd"), board)
        pair_sevens = evaluate(_cards("7c", "7d"), board)
        assert pair_kings > pair_sevens

    def test_compare_hands_single_winner(self):
        board = _cards("2c", "5d", "8h", "Js", "3c")
        hand_a = _cards("Ah", "Kh")  # ace high
        hand_b = _cards("7c", "7d")  # pair of sevens
        winners = compare_hands([hand_a, hand_b], board)
        assert winners == [1]

    def test_compare_hands_tie(self):
        board = _cards("2c", "5d", "8h", "Js", "Ac")
        hand_a = _cards("Kh", "3h")
        hand_b = _cards("Kd", "3d")
        winners = compare_hands([hand_a, hand_b], board)
        assert winners == [0, 1]


# ── Pot ─────────────────────────────────────────────────────────────────

class TestPot:
    def test_add_and_total(self):
        pot = Pot()
        pot.add(0, 50)
        pot.add(1, 50)
        assert pot.total == 100

    def test_round_bet_tracking(self):
        pot = Pot()
        pot.add(0, 20)
        pot.add(1, 20)
        assert pot.player_round_bet(0) == 20
        assert pot.player_round_bet(1) == 20
        pot.new_round()
        assert pot.player_round_bet(0) == 0

    def test_settle_single_winner(self):
        pot = Pot()
        pot.add(0, 100)
        pot.add(1, 100)
        payouts = pot.settle([0])
        assert payouts == {0: 200}

    def test_settle_split_pot(self):
        pot = Pot()
        pot.add(0, 100)
        pot.add(1, 100)
        payouts = pot.settle([0, 1])
        assert payouts[0] + payouts[1] == 200
        assert payouts[0] == 100

    def test_reset(self):
        pot = Pot()
        pot.add(0, 50)
        pot.reset()
        assert pot.total == 0

    def test_side_pot_all_in(self):
        pot = Pot()
        pot.add(0, 100)
        pot.add(1, 200)
        pot.build_side_pots(
            all_in_amounts={0: 100, 1: 200},
            active_ids={0, 1},
        )
        assert len(pot.side_pots) == 2
        assert pot.side_pots[0]["amount"] == 200
        assert pot.side_pots[0]["eligible"] == {0, 1}
        assert pot.side_pots[1]["amount"] == 100
        assert pot.side_pots[1]["eligible"] == {1}


# ── Game ────────────────────────────────────────────────────────────────

def _always_call(state: GameState):
    """Simple player that always calls or checks."""
    to_call = state.current_bet - state.player_round_bet
    if to_call > 0:
        return (Action.CALL, 0)
    return (Action.CHECK, 0)


def _always_fold(state: GameState):
    """Player that always folds (unless can check)."""
    to_call = state.current_bet - state.player_round_bet
    if to_call > 0:
        return (Action.FOLD, 0)
    return (Action.CHECK, 0)


class TestGame:
    def test_play_hand_completes(self):
        game = Game(stacks=(1000, 1000), deck_seed=42)
        result = game.play_hand(_always_call)
        assert "winners" in result
        assert "payouts" in result
        assert len(result["community_cards"]) == 5

    def test_chips_conserved(self):
        game = Game(stacks=(1000, 1000), deck_seed=42)
        game.play_hand(_always_call)
        total = sum(p.chips for p in game.players)
        assert total == 2000

    def test_fold_ends_hand(self):
        game = Game(stacks=(1000, 1000), deck_seed=42)

        def fold_on_second(state):
            if state.player_id == 1:
                return (Action.FOLD, 0)
            return (Action.CALL, 0)

        result = game.play_hand(fold_on_second)
        assert 0 in result["winners"]
        assert game.players[0].chips > 1000

    def test_blinds_posted(self):
        game = Game(stacks=(1000, 1000), small_blind=10, big_blind=20, deck_seed=42)
        game.play_hand(_always_call)
        assert game.pot.total == 0  # pot settled
        total = sum(p.chips for p in game.players)
        assert total == 2000

    def test_advance_dealer(self):
        game = Game(stacks=(1000, 1000), deck_seed=42)
        assert game.dealer == 0
        game.advance_dealer()
        assert game.dealer == 1
        game.advance_dealer()
        assert game.dealer == 0

    def test_multiple_hands(self):
        game = Game(stacks=(1000, 1000), deck_seed=42)
        for _ in range(10):
            game.play_hand(_always_call)
            game.advance_dealer()
        total = sum(p.chips for p in game.players)
        assert total == 2000

    def test_raise_action(self):
        game = Game(stacks=(1000, 1000), deck_seed=42)

        def raise_once(state):
            if state.street == Street.PREFLOP and state.player_round_bet < 40:
                return (Action.RAISE, 60)
            to_call = state.current_bet - state.player_round_bet
            if to_call > 0:
                return (Action.CALL, 0)
            return (Action.CHECK, 0)

        result = game.play_hand(raise_once)
        assert "winners" in result
        total = sum(p.chips for p in game.players)
        assert total == 2000

    def test_all_in(self):
        game = Game(stacks=(100, 100), small_blind=10, big_blind=20, deck_seed=42)

        def go_all_in(state):
            return (Action.ALL_IN, state.player_chips)

        result = game.play_hand(go_all_in)
        assert "winners" in result
        total = sum(p.chips for p in game.players)
        assert total == 200

    def test_game_state_has_hole_cards(self):
        game = Game(stacks=(1000, 1000), deck_seed=42)
        states = []

        def capture_state(state):
            states.append(state)
            to_call = state.current_bet - state.player_round_bet
            if to_call > 0:
                return (Action.CALL, 0)
            return (Action.CHECK, 0)

        game.play_hand(capture_state)
        assert len(states) > 0
        for s in states:
            assert len(s.hole_cards) == 2
            assert s.player_chips >= 0

    def test_unequal_stacks(self):
        game = Game(stacks=(500, 1500), deck_seed=42)
        game.play_hand(_always_call)
        total = sum(p.chips for p in game.players)
        assert total == 2000

    def test_all_in_no_action_on_later_streets(self):
        """After all-in + call, remaining streets should deal without prompting."""
        game = Game(stacks=(100, 1000), small_blind=10, big_blind=20, deck_seed=42)
        action_counts = {0: 0, 1: 0}

        def track_actions(state):
            action_counts[state.player_id] += 1
            if state.player_id == 0:
                return (Action.ALL_IN, state.player_chips)
            to_call = state.current_bet - state.player_round_bet
            if to_call > 0:
                return (Action.CALL, 0)
            return (Action.CHECK, 0)

        result = game.play_hand(track_actions)
        # Player 0 goes all-in preflop, player 1 calls.
        # No further actions should be prompted on flop/turn/river.
        assert action_counts[0] <= 1  # only the all-in
        assert action_counts[1] <= 1  # only the call
        assert len(result["community_cards"]) == 5
        total = sum(p.chips for p in game.players)
        assert total == 1100

    def test_both_all_in_runs_out_board(self):
        """Both players all-in preflop should deal all 5 community cards."""
        game = Game(stacks=(100, 100), small_blind=10, big_blind=20, deck_seed=42)

        def always_all_in(state):
            return (Action.ALL_IN, state.player_chips)

        result = game.play_hand(always_all_in)
        assert len(result["community_cards"]) == 5
        assert "winners" in result
        total = sum(p.chips for p in game.players)
        assert total == 200

    def test_short_stack_blind_all_in(self):
        """Player whose stack <= big blind is all-in from the blind."""
        game = Game(stacks=(15, 1000), small_blind=10, big_blind=20, deck_seed=42)
        action_counts = {0: 0, 1: 0}

        def track(state):
            action_counts[state.player_id] += 1
            to_call = state.current_bet - state.player_round_bet
            if to_call > 0:
                return (Action.CALL, 0)
            return (Action.CHECK, 0)

        result = game.play_hand(track)
        assert len(result["community_cards"]) == 5
        total = sum(p.chips for p in game.players)
        assert total == 1015
