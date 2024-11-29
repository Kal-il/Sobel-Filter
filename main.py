import cv2 as cv
import numpy as np


def to_grayscale(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def sobel_filter(image):

    # definição das mascaras:
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    rows, cols = image.shape

    Gx = np.zeros_like(image)
    Gy = np.zeros_like(image)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            Gx[i, j] = np.sum(Kx * image[i - 1:i + 2, j - 1:j + 2])
            Gy[i, j] = np.sum(Ky * image[i - 1:i + 2, j - 1:j + 2])

    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G = (G / G.max()) * 255

    return G.astype(np.uint8)


image = cv.imread("taj_orig.jpg")
gray_image = to_grayscale(image)
sobel_image = sobel_filter(gray_image)

# Exibir a imagem
cv.imshow("Sobel Filter", sobel_image)
cv.waitKey(0)
cv.destroyAllWindows()