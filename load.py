from models.matrix import Matrix
from models.vector import Vector
from utils.dsv1 import read_dsv1

images, labels = read_dsv1("data/mnist_train.dsv1")


def images_to_matrix(img) -> Matrix:
    matrix = []
    for img in images:
        flat = img.reshape(-1)      # flatten
        matrix.append(flat.tolist())  # numpy -> python list
    return Matrix(matrix)

def labels_to_vector(l) -> Vector:
    return Vector(labels.tolist())


X = images_to_matrix(images)

y = labels_to_vector(labels)

