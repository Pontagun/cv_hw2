import cv2
import numpy as np
from PIL import Image
import math
import main as m
from statistics import mean

# i = [[1, 1, 1, 1, 1],
#      [1, 1, 1, 1, 1],
#      [1, 1, 1, 1, 1],
#      [1, 1, 1, 1, 1],
#      [1, 1, 1, 1, 1]]
# a = [[1, -2, 2, -2, 1],
#      [1, -2, 2, -2, 1],
#      [1, -2, 2, -2, 1],
#      [1, -2, 2, -2, 1],
#      [1, -2, 2, -2, 1]]
# b = [[-1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1],
#      [4, 4, 4, 4, 4],
#      [-1, -1, -1, -1, -1],
#      [-1, -1, -1, -1, -1]]
# c = [[-1, 2, -4, 4, -1],
#      [-1, 2, -4, 4, -1],
#      [-1, 2, -4, 4, -1],
#      [-1, 2, -4, 4, -1],
#      [1, 1, -4, 1, 1]]
# d = [[1, 1, -2, 1, 1],
#      [1, 1, -2, 1, 1],
#      [-1, -2, -2, -2, -1],
#      [1, 1, -2, 1, 1],
#      [1, 1, -2, 1, 1]]
# e = [[-2, 1, 2, 1, -2],
#      [1, -2, 1, -2, 1],
#      [1, 1, -2, 1, 1],
#      [1, -2, 1, -2, 1],
#      [-2, 1, 2, 1, -2]]
# f = [[-4, 1, 1, 1, -4],
#      [1, -4, 1, -4, 1],
#      [1, 1, -4, 1, 1],
#      [1, 1, 1, 1, 1],
#      [1, 1, 1, 1, 1]]
# i = [[1, 1, 1, 1, -4],
#      [1, 1, 1, -4, 1],
#      [1, 1, -4, 1, 1],
#      [1, 1, -4, 1, 1],
#      [1, 1, -4, 1, 1]]
# j = [[1, 1, -4, 1, 1],
#        [1, 1, -4, 1, 1],
#        [1, 1, -4, 1, 1],
#        [1, -4, 1, 1, 1],
#        [-4, 1, 1, 1, 1]]
#
# image = cv2.imread('pic5.jpg', 0)
# iw = np.zeros((image.shape[0], image.shape[1]))
#
# identity = cv2.filter2D(src=image, ddepth=-1, kernel=np.array(j))
#
# identity = m.pixel_log255(identity)
#
# identity = identity.flatten()
#
# for i, val in enumerate(identity):
#     if val > 50:
#         identity[i] = 255
#     else:
#         identity[i] = 0
#
# grad = np.reshape(identity, (image.shape[0], image.shape[1]))
# im = Image.fromarray(grad)
# if im.mode != 'RGB':
#     im = im.convert('RGB')
# im.save("j.jpg")

image_aa = cv2.imread('a.jpg', 0)
image_bb = cv2.imread('b.jpg', 0)
image_cc = cv2.imread('c.jpg', 0)
image_dd = cv2.imread('d.jpg', 0)
image_ee = cv2.imread('e.jpg', 0)
image_ff = cv2.imread('f.jpg', 0)
image_ii = cv2.imread('i.jpg', 0)
image_jj = cv2.imread('j.jpg', 0)

iw_all = np.zeros((image_aa.shape[0], image_aa.shape[1]))

for i in range(image_aa.shape[0]):
    for j in range(image_aa.shape[1]):
        numbers = [image_aa[i][j], image_bb[i][j], image_cc[i][j], image_dd[i][j], image_ee[i][j], image_ff[i][j], image_ii[i][j], image_jj[i][j]]
        iw_all[i][j] = mean(numbers)

im = Image.fromarray(iw_all)
if im.mode != 'RGB':
    im = im.convert('RGB')
im.save("mean8degcl.jpg")
