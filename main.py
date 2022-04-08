import cv2
import numpy as np
from PIL import Image
import math


def pixel_log255(unorm_image):
    pxmin = unorm_image.min()
    pxmax = unorm_image.max()

    for i in range(unorm_image.shape[0]):
        for j in range(unorm_image.shape[1]):
            # unorm_image[i, j] = (255 / math.log10(256)) * math.log10(1 + (255 / pxmax) * unorm_image[i, j])
            unorm_image[i, j] = ((unorm_image[i, j] - pxmin) / (pxmax - pxmin)) * 255

    norm_image = unorm_image
    return norm_image


def iLoG(shape, std):
    s = (shape, shape)
    l_o_g = np.zeros(s)

    a = 1 / (2 * math.pi * (std ** 4))

    lim = int(math.floor(shape / 2))

    for row in range(-lim, lim + 1):
        for col in range(-lim, lim + 1):
            b = ((row ** 2) + (col ** 2) - (2 * (std ** 2))) / (std ** 2)
            c = np.exp((-1 / (2 * (std ** 2))) * (row ** 2 + col ** 2))
            l_o_g[row + lim][col + lim] = a * b * c

    return l_o_g


def sobel():
    wx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    wy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    return wx, wy


def innerprod(img, w):
    iw = np.zeros((img.shape[0], img.shape[1]))

    for r in range(1, len(img) - 1):
        for c in range(1, len(img) - 1):
            iw[r][c] = w[0][0] * img[r - 1][c - 1] \
                       + w[0][1] * img[r - 1][c] \
                       + w[0][2] * img[r - 1][c + 1] \
                       + w[1][0] * img[r][c - 1] \
                       + w[1][1] * img[r][c] \
                       + w[1][2] * img[r][c + 1] \
                       + w[2][0] * img[r + 1][c - 1] \
                       + w[2][1] * img[r + 1][c] \
                       + w[2][2] * img[r + 1][c + 1]

    return iw


if __name__ == '__main__':
    image = cv2.imread('bank 256x256.jpg', 0)
    iw = np.zeros((image.shape[0], image.shape[1]))

    for r in range(1, len(image) - 1):
        for c in range(1, len(image) - 1):
            kirsh_order = []
            kirsh_order.append(int(image[r - 1][c - 1]))
            kirsh_order.append(int(image[r - 1][c]))
            kirsh_order.append(int(image[r - 1][c + 1]))
            kirsh_order.append(int(image[r][c + 1]))
            kirsh_order.append(int(image[r + 1][c + 1]))
            kirsh_order.append(int(image[r + 1][c]))
            kirsh_order.append(int(image[r + 1][c - 1]))
            kirsh_order.append(int(image[r][c - 1]))

            g = []

            for i in range(8):
                s = kirsh_order[i % 8] + kirsh_order[(i + 1) % 8] + kirsh_order[(i + 2) % 8]
                t = kirsh_order[(i + 3) % 8] + kirsh_order[(i + 4) % 8] + kirsh_order[(i + 5) % 8] \
                    + kirsh_order[(i + 6) % 8] + kirsh_order[(i + 7) % 8]

                g.append(abs(5 * s - 3 * t))

            iw[r][c] = max(g)

    grad = pixel_log255(iw)
    grad = grad.flatten()

    for i, val in enumerate(grad):
        if val > 50:
            grad[i] = 255
        else:
            grad[i] = 0

    grad = np.reshape(grad, (image.shape[0], image.shape[1]))
    im = Image.fromarray(grad)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("kirsh.jpg")
