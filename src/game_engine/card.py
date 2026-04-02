"""Card and Deck classes for Texas Hold'em, backed by the treys library."""

from __future__ import annotations

from treys import Card as TreysCard
from treys import Deck as TreysDeck


# Treys uses integer representations internally.  We expose thin wrappers
# so the rest of the codebase can work with readable card objects while
# still converting to/from treys ints for evaluation.

# Valid rank characters accepted by treys
RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "shdc"  # spades, hearts, diamonds, clubs


class Card:
    """Thin wrapper around a treys card integer.

    Create with a two-character string like ``Card("As")`` for Ace of spades,
    or from a raw treys int with ``Card.from_int(i)``.
    """

    __slots__ = ("_int",)

    def __init__(self, card_str: str):
        self._int: int = TreysCard.new(card_str)

    @classmethod
    def from_int(cls, treys_int: int) -> Card:
        """Build a Card directly from a treys integer."""
        c = object.__new__(cls)
        c._int = treys_int
        return c

    def to_int(self) -> int:
        """Return the treys integer representation."""
        return self._int

    @property
    def rank_char(self) -> str:
        return TreysCard.STR_RANKS[TreysCard.get_rank_int(self._int)]

    @property
    def suit_char(self) -> str:
        suit_int = TreysCard.get_suit_int(self._int)
        suit_map = {1: "s", 2: "h", 4: "d", 8: "c"}
        return suit_map[suit_int]

    def __repr__(self):
        return TreysCard.int_to_str(self._int)

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self._int == other._int

    def __hash__(self):
        return hash(self._int)

    def __lt__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return TreysCard.get_rank_int(self._int) < TreysCard.get_rank_int(other._int)


class Deck:
    """A standard 52-card deck backed by treys.

    Args:
        seed: Not used by treys (it uses ``random.shuffle`` internally),
              but accepted for API compatibility.  For reproducible deals
              set ``random.seed()`` before constructing the deck.
    """

    def __init__(self, seed=None):
        if seed is not None:
            import random
            random.seed(seed)
        self._deck = TreysDeck()

    def reset(self):
        """Rebuild and shuffle the deck."""
        self._deck = TreysDeck()

    def deal(self, n: int = 1) -> list[Card]:
        """Deal n cards from the deck."""
        ints = self._deck.draw(n)
        if isinstance(ints, int):
            ints = [ints]
        return [Card.from_int(i) for i in ints]

    def __len__(self):
        return len(self._deck.cards)
