from __future__ import annotations


import math
from functools import lru_cache


@lru_cache
def sum_denominator(vector: "Vector") -> float:
    return sum([math.exp(value) for value in vector])


class Softmax:

    @staticmethod
    def forward(x: float, vector: "Vector") -> float:
        exp_sum = sum_denominator(vector)
        return math.exp(x) / exp_sum