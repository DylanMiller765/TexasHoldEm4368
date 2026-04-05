class Evaluator:

    def __init__(self):
        self.hands       = 0
        self.human_wins  = 0
        self.ai_wins     = 0
        self.ties        = 0
        self.chip_diff_history: list[int] = []  # human chips - 1000 after each hand
        self.win_history: list[float]     = []  # human win rate % after each hand

    def record(self, winner: str, human_chips: int, ai_chips: int):
        self.hands += 1

        if winner == "human":
            self.human_wins += 1
        elif winner == "ai":
            self.ai_wins += 1
        else:
            self.ties += 1

        self.chip_diff_history.append(human_chips - ai_chips)
        self.win_history.append(self.win_rate)

    @property
    def win_rate(self) -> float:
        if self.hands == 0:
            return 0.0
        return round((self.human_wins / self.hands) * 100, 1)

    @property
    def chip_differential(self) -> int:
        if not self.chip_diff_history:
            return 0
        return self.chip_diff_history[-1]

    def summary(self) -> dict:
        return \
        {
            "hands":            self.hands,
            "human_wins":       self.human_wins,
            "ai_wins":          self.ai_wins,
            "ties":             self.ties,
            "win_rate":         self.win_rate,
            "chip_differential": self.chip_differential,
            "win_history":      self.win_history[-100:],  # last 100 for chart
        }

    def reset(self):
        """Reset all stats."""
        self.__init__()