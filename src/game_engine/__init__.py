"""Texas Hold'em game engine.

Public API:
    - Card, Deck: Card representations (backed by treys).
    - HandCategory, HandRank, evaluate, compare_hands: Hand evaluation.
    - Pot: Pot and side-pot management.
    - Game, GameState, Action, Street, Player: Game orchestration.
"""

from .card import Card, Deck
from .hand_eval import HandCategory, HandRank, evaluate, compare_hands
from .pot import Pot
from .game import Game, GameState, Action, Street, Player

__all__ = [
    "Card", "Deck",
    "HandCategory", "HandRank", "evaluate", "compare_hands",
    "Pot",
    "Game", "GameState", "Action", "Street", "Player",
]
