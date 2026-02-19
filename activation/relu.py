
class ReLU:

    @staticmethod
    def forward(x: float) -> float:
        if x > 0.0:
            return x
        return 0

    @staticmethod
    def backward(x: float):
        if x > 0.0:
            return 1
        return 0