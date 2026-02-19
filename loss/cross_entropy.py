import math

from core.errors import VectorError
from vector import Vector


class CrossEntropy:

    @staticmethod
    def loss(p: Vector, y: Vector):

        if p.size != y.size:
            raise VectorError("The two vectors must have the same size")

        r = - sum([y[i] * math.log(p[i]) for i in range(p.size)])

        return r