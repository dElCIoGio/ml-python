from __future__ import annotations


from core.errors import MatrixSumError, MatrixMultiplicationError
from protocols.activation_function import ActivationFn
from vector import Vector


class Matrix:
    def __init__(self, matrix: list[list[float] | Vector]) -> None:

        self._cols = len(matrix[0])
        self._rows = len(matrix)

        self.matrix = []
        for row in matrix:
            self.matrix.append(Vector(row))

    @staticmethod
    def zeros(cols: int, rows: int) -> Matrix:
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])

    @staticmethod
    def T(matrix: Matrix) -> Matrix:
        m = []
        for col in range(matrix.columns):
            new_row = []
            for row in range(matrix.rows):
                new_row.append(matrix[row][col])
            m.append(new_row)
        return Matrix(m)

    @property
    def columns(self) -> int:
        return self._cols

    @property
    def rows(self) -> int:
        return self._rows

    @staticmethod
    def multiply(a: Matrix, b: Matrix, *, transpose_a: bool = False, transpose_b: bool = False) -> Matrix:

        if transpose_a:
            a = Matrix.T(a)

        if transpose_b:
            b = Matrix.T(b)

        return a * b

    def activate(self, activation_fn: ActivationFn):
        for r in range(self.rows):
            self.matrix[r].activate(activation_fn=activation_fn)


    def fill(self, value: float) -> None:
        for c in range(self._cols):
            for r in range(self._rows):
                self.matrix[c][r] = value

    def scale(self, value: float) -> None:
        for c in range(self._cols):
            for r in range(self._rows):
                self.matrix[c][r] *= value

    def sum(self) -> float:

        total = 0.0
        for c in range(self._cols):
            for r in range(self._rows):
                total += self.matrix[c][r]

        return total

    def __getitem__(self, index: int) -> Vector:
        return self.matrix[index]

    def __copy__(self):
        return Matrix(self.matrix)

    def __repr__(self) -> str:
        out = ""
        for i in range(self.rows):
            r = self.matrix[i]
            out += f"{r.__str__()}\n"
        return out

    def __add__(self, other: Matrix) -> Matrix:
        if self.rows != other.rows or self.columns != other.columns:
            raise MatrixSumError(f"Incompatible shapes: {self.rows}X{self.columns} and {other.rows}X{other.columns}")

        result = []

        for c in range(self.columns):
            row = []
            for r in range(self.rows):
                row.append(self.matrix[c][r] + other.matrix[c][r])
            result.append(row)

        return Matrix(result)

    def __sub__(self, other: Matrix) -> Matrix:
        if self.rows != other.rows or self.columns != other.columns:
            raise MatrixSumError(f"Incompatible shapes: {self.rows}X{self.columns} and {other.rows}X{other.columns}")

        result = []

        for c in range(self.columns):
            row = []
            for r in range(self.rows):
                row.append(self.matrix[c][r] - other.matrix[c][r])
            result.append(row)

        return Matrix(result)


    def __mul__(self, other: Matrix) -> Matrix:

        if self.columns != other.rows:
            raise MatrixMultiplicationError("Inner dimensions must match")

        new_matrix = []
        other = self.T(other)

        for r1 in range(self.rows):
            new_row = []
            for r2 in range(other.rows):
                new_row.append(self.matrix[r1].dot(other.matrix[r2]))
            new_matrix.append(new_row)

        return Matrix(new_matrix)


