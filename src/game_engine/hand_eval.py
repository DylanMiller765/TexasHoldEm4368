"""Hand ranking and evaluation for Texas Hold'em, backed by treys.

Treys uses a lookup-table evaluator that is much faster than brute-force
combinatorial evaluation.  Lower rank numbers are *better* hands in treys
(1 = Royal Flush).  We wrap this so that comparison operators work
intuitively (higher HandRank = better hand).
"""

from __future__ import annotations

from enum import IntEnum

from treys import Evaluator as TreysEvaluator

from .card import Card

# Singleton evaluator — the lookup tables are expensive to build, so we
# only do it once.
_evaluator = TreysEvaluator()


class HandCategory(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8

# Map treys rank class (1-9) to our HandCategory.
# Treys: 1=Straight Flush, 2=Four of a Kind, ... 9=High Card
_TREYS_CLASS_TO_CATEGORY = {
    0: HandCategory.STRAIGHT_FLUSH,  # Royal flush
    1: HandCategory.STRAIGHT_FLUSH,
    2: HandCategory.FOUR_OF_A_KIND,
    3: HandCategory.FULL_HOUSE,
    4: HandCategory.FLUSH,
    5: HandCategory.STRAIGHT,
    6: HandCategory.THREE_OF_A_KIND,
    7: HandCategory.TWO_PAIR,
    8: HandCategory.ONE_PAIR,
    9: HandCategory.HIGH_CARD,
}


class HandRank:
    """Comparable hand ranking backed by treys evaluation.

    Attributes:
        category: The HandCategory (e.g. FLUSH, TWO_PAIR).
        treys_rank: The raw treys rank (lower = better).
        cards: The cards that were evaluated.
    """

    __slots__ = ("category", "treys_rank", "cards")

    def __init__(self, treys_rank: int, cards: list[Card]):
        self.treys_rank = treys_rank
        rank_class = _evaluator.get_rank_class(treys_rank)
        self.category = _TREYS_CLASS_TO_CATEGORY[rank_class]
        self.cards = cards

    @property
    def _key(self):
        # Lower treys_rank = better hand, so negate for natural ordering
        return (-self.treys_rank,)

    def __eq__(self, other):
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.treys_rank == other.treys_rank

    def __lt__(self, other):
        if not isinstance(other, HandRank):
            return NotImplemented
        # Lower treys rank is better, so "less than" means higher treys number
        return self.treys_rank > other.treys_rank

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.treys_rank < other.treys_rank

    def __ge__(self, other):
        return self == other or self > other

    def __repr__(self):
        class_str = _evaluator.class_to_string(
            _evaluator.get_rank_class(self.treys_rank)
        )
        return f"{class_str} (rank={self.treys_rank})"


def evaluate(hole_cards: list[Card], community_cards: list[Card]) -> HandRank:
    """Evaluate a player's best hand from hole cards + community cards.

    Args:
        hole_cards: The player's 2 private cards.
        community_cards: Exactly 5 community cards.

    Returns:
        A HandRank object (comparable with >, <, ==).
    """
    hand_ints = [c.to_int() for c in hole_cards]
    board_ints = [c.to_int() for c in community_cards]
    treys_rank = _evaluator.evaluate(board_ints, hand_ints)
    return HandRank(treys_rank, hole_cards + community_cards)


def compare_hands(
    hands: list[list[Card]],
    community_cards: list[Card],
) -> list[int]:
    """Compare multiple players' hands and return winner indices.

    Args:
        hands: List of [hole_card_1, hole_card_2] per player.
        community_cards: The 5 community cards.

    Returns:
        List of player indices that tied for the best hand.
    """
    rankings = [evaluate(h, community_cards) for h in hands]
    best = max(rankings)
    return [i for i, r in enumerate(rankings) if r == best]
