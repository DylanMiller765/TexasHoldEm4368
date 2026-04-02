"""Pot and side-pot management for Texas Hold'em."""


class Pot:
    """Manages the main pot and any side pots created by all-in situations.

    In a two-player game side pots are rare (only when one player is all-in
    for less than the other's bet), but we handle the general case so the
    engine scales if needed.
    """

    def __init__(self):
        self.main_pot: int = 0
        self.side_pots: list[dict] = []  # [{"amount": int, "eligible": set[int]}]
        # Per-round contribution tracking (player_id -> chips put in this round)
        self._round_bets: dict[int, int] = {}

    @property
    def total(self) -> int:
        """Total chips across main pot and all side pots."""
        return self.main_pot + sum(sp["amount"] for sp in self.side_pots)

    def add(self, player_id: int, amount: int):
        """Add chips to the pot from a player."""
        self.main_pot += amount
        self._round_bets[player_id] = self._round_bets.get(player_id, 0) + amount

    def player_round_bet(self, player_id: int) -> int:
        """How much a player has bet in the current betting round."""
        return self._round_bets.get(player_id, 0)

    def new_round(self):
        """Reset per-round bet tracking (called between betting rounds)."""
        self._round_bets.clear()

    def build_side_pots(self, all_in_amounts: dict[int, int], active_ids: set[int]):
        """Build side pots when one or more players are all-in.

        Args:
            all_in_amounts: Map of player_id -> total chips contributed to pot
                this hand (across all rounds) for players who are all-in.
            active_ids: Set of all player ids still in the hand (not folded).
        """
        if not all_in_amounts:
            return

        # Gather total contributions for all active players
        # Players not all-in are assumed to have matched the highest bet
        max_contrib = max(all_in_amounts.values()) if all_in_amounts else 0
        contributions = dict(all_in_amounts)
        for pid in active_ids:
            if pid not in contributions:
                contributions[pid] = max_contrib

        sorted_amounts = sorted(set(contributions.values()))
        pots = []
        prev = 0
        for level in sorted_amounts:
            diff = level - prev
            eligible = {pid for pid, amt in contributions.items() if amt >= level}
            pot_amount = diff * len(eligible)
            pots.append({"amount": pot_amount, "eligible": eligible})
            prev = level

        self.side_pots = pots
        self.main_pot = 0  # All chips moved into side pots

    def settle(self, winner_ids: list[int]) -> dict[int, int]:
        """Distribute pot(s) to winners.

        Args:
            winner_ids: Player indices that won (ties possible).

        Returns:
            Dict mapping player_id -> chips awarded.
        """
        payouts: dict[int, int] = {}

        if self.side_pots:
            for sp in self.side_pots:
                eligible_winners = [w for w in winner_ids if w in sp["eligible"]]
                if not eligible_winners:
                    # If no winner is eligible, give to all eligible players
                    # (shouldn't happen in normal play, but safe fallback)
                    eligible_winners = list(sp["eligible"])
                share = sp["amount"] // len(eligible_winners)
                remainder = sp["amount"] % len(eligible_winners)
                for i, pid in enumerate(eligible_winners):
                    award = share + (1 if i < remainder else 0)
                    payouts[pid] = payouts.get(pid, 0) + award
            self.side_pots.clear()
        else:
            share = self.main_pot // len(winner_ids)
            remainder = self.main_pot % len(winner_ids)
            for i, pid in enumerate(winner_ids):
                award = share + (1 if i < remainder else 0)
                payouts[pid] = payouts.get(pid, 0) + award
            self.main_pot = 0

        return payouts

    def reset(self):
        """Clear all pot state for a new hand."""
        self.main_pot = 0
        self.side_pots.clear()
        self._round_bets.clear()
