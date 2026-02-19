from typing import Protocol

from models.vector import Vector


class LossFn(Protocol):

    @staticmethod
    def loss(x: Vector, y: Vector, **kwargs):
        ...