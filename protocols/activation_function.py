

from typing import Protocol, Optional


class ActivationFn(Protocol):

    @staticmethod
    def forward(x: float, **kwargs) -> float:
        ...

    @staticmethod
    def backward(x: float, **kwargs) -> float:
        ...