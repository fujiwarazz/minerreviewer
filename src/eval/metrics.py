from __future__ import annotations

from statistics import mean

from common.types import ArbiterOutput


def strength_weakness_balance(output: ArbiterOutput) -> float:
    total = len(output.strengths) + len(output.weaknesses)
    if total == 0:
        return 0.0
    return len(output.strengths) / total


def average_rating(outputs: list[ArbiterOutput]) -> float:
    ratings = [output.raw_rating for output in outputs if output.raw_rating is not None]
    return mean(ratings) if ratings else 0.0
