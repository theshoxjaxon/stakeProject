"""Value detection utilities (margin removal, edge, Kelly staking)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ValueQuant:
    """Tools for transforming market odds into fair prices and stakes."""

    kelly_fraction: float = 0.25

    def remove_margin(self, odds: Iterable[float]) -> List[float]:
        """
        Remove bookmaker margin using an additive normalization scheme.

        For prices odds_i, raw implied probabilities are p_i = 1/odds_i.
        We find c such that sum(max(p_i - c, 0)) == 1, then define
        fair odds as 1 / (p_i - c).
        """
        ps = [1.0 / o for o in odds if o and o > 0]
        if not ps:
            return list(odds)

        lo, hi = 0.0, min(ps) - 1e-9

        def total(c: float) -> float:
            return sum(max(p - c, 0.0) for p in ps)

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if total(mid) > 1.0:
                lo = mid
            else:
                hi = mid

        c = 0.5 * (lo + hi)
        fair: List[float] = []
        for o in odds:
            if not o or o <= 0:
                fair.append(o)
                continue
            p_raw = 1.0 / o
            p_adj = max(p_raw - c, 1e-9)
            fair.append(1.0 / p_adj)
        return fair

    def calculate_stake(self, edge: float, odds: float) -> float:
        """
        Fractional Kelly stake as a fraction of bankroll.

        Assumes our true probability is p = implied + edge, where
        implied = 1 / odds and edge = p - implied.
        """
        if odds <= 0:
            return 0.0
        implied = 1.0 / odds
        p = implied + edge
        if p <= 0.0 or p >= 1.0:
            return 0.0
        q = 1.0 - p
        b = odds - 1.0
        if b <= 0:
            return 0.0
        full_kelly = (b * p - q) / b
        stake = self.kelly_fraction * full_kelly
        return max(0.0, min(1.0, stake))

