from utils.dsv1 import read_dsv1
import matplotlib.pyplot as plt


def display_img(filename: str, index: int = 0, h: int = 28, w: int = 28):
    images, labels = read_dsv1(filename)

    img = images[index].reshape(h, w)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {labels[0]}")
    plt.axis("off")
    plt.show()


def display_images(filename: str, figsize: int = 6):
    
    images, labels = read_dsv1(filename)

    plt.figure(figsize=(figsize,figsize))
    
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].reshape(28,28), cmap="gray")
        plt.title(labels[i])
        plt.axis("off")
    
    plt.show()