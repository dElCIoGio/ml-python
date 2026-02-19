import math

class Sigmoid:

    @staticmethod
    def forward(x: float) -> float:
        return 1/(1 + math.exp(-x))

    @staticmethod
    def backward(x: float) -> float:
        return Sigmoid.forward(x) * (1 - Sigmoid.forward(x))