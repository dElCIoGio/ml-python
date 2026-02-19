

class MLError(Exception):
    pass



# Vector Errors

class VectorError(MLError):
    pass


class VectorDotError(VectorError):
    pass


class MatrixError(MLError):
    pass

class MatrixSumError(MatrixError):
    pass

class MatrixMultiplicationError(MatrixError):
    pass


# Loss Function Errors

class LossError(MLError):
    pass

