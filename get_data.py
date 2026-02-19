import tensorflow_datasets as tfds
import numpy as np
from utils.dsv1 import write_dsv1


def ds_to_numpy(ds):

    images = []
    labels = []

    for image, label in ds:
        images.append(image.numpy())
        labels.append(label.numpy())

    return np.array(images), np.array(labels)

train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

train_images, train_labels = ds_to_numpy(train_ds)
test_images, test_labels = ds_to_numpy(test_ds)

train_images = train_images.astype(np.uint8)
test_images = test_images.astype(np.uint8)

train_labels = train_labels.astype(np.uint8)
test_labels = test_labels.astype(np.uint8)

write_dsv1("data/mnist_train.dsv1", train_images, train_labels)
write_dsv1("data/mnist_test.dsv1", test_images, test_labels)


print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# (60000, 28, 28, 1)
# (60000,)
# (10000, 28, 28, 1)
# (10000,)


#
# import numpy as np
# from utils.dsv1 import read_dsv1
#
# train_images, train_labels = read_dsv1("data/mnist_train.dsv1")
# test_images, test_labels = read_dsv1("data/mnist_test.dsv1")
#
# train_images = train_images.astype(np.float32) / 255.0
# test_images  = test_images.astype(np.float32) / 255.0
#
# train_x = train_images.reshape(train_images.shape[0], -1)
# test_x  = test_images.reshape(test_images.shape[0], -1)
#
# train_y = train_labels.astype(np.int32)
# test_y  = test_labels.astype(np.int32)
#
# print(train_x.shape, train_y.shape)
# print(test_x.shape, test_y.shape)
#
#
#
