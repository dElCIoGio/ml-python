from loss.cross_entropy import CrossEntropy
from matrix import Matrix

p = Matrix([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

y = Matrix([[1, 0, 0], [0, 1, 0]])

loss = CrossEntropy()

print(p.loss(y, loss))
