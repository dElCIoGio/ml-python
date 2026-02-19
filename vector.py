
from __future__ import annotations

from functools import cached_property

from core.errors import VectorDotError
from protocols.activation_function import ActivationFn


class Vector:
    def __init__(self, data: list[float]):

        self.data = data
        self._size = len(self.data)


    @cached_property
    def size(self) -> int:
        return self._size

    @staticmethod
    def zeros(size: int) -> Vector:
        return Vector([0] * size)

    def dot(self, other: Vector) -> float:

        if self.size != other.size:
            # TODO: write error message
            raise VectorDotError

        total = 0.0
        for i in range(self.size):
            total += self.data[i] * other.data[i]

        return total

    def activate(self, activation_fn: ActivationFn) -> None:

        for i in range(self.size):
            self.data[i] = activation_fn.forward(self.data[i], vector=self)

    def __iter__(self):
        for row in self.data:
            yield row

    def __add__(self, other: Vector) -> Vector:
        result = [self.data[i] + other.data[i] for i, _ in enumerate(self.__iter__())]
        return Vector(result)

    def __sub__(self, other: Vector) -> Vector:
        result = [self.data[i] - other.data[i] for i, _ in enumerate(self.__iter__())]
        return Vector(result)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    def __delitem__(self, index: int) -> None:
        del self.data[index]

    def __repr__(self):
        out = ""
        for val in self.__iter__():
            out += f"{val:.3f}"
            out += "\t"
        return out.strip()