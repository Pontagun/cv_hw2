import cv2
import numpy as np
from PIL import Image
import math
import main as m
from statistics import mean

i = [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]]
a = [[1, 1, 1, 1, 1 * (-4)],
     [1, 1, 1, 1 * (-4), 1],
     [1, 1, 1 * (-4), 1, 1],
     [1, 1 * (-4), 1, 1, 1],
     [1 * (-4), 1, 1, 1, 1]]
b = [[1 * (-4), 1, 1, 1, 1],
     [1, 1 * (-4), 1, 1, 1],
     [1, 1, 1 * (-4), 1, 1],
     [1, 1, 1, 1 * (-4), 1],
     [1, 1, 1, 1, 1 * (-4)]]
c = [[1, 1, 1 * (-4), 1, 1],
     [1, 1, 1 * (-4), 1, 1],
     [1, 1, 1 * (-4), 1, 1],
     [1, 1, 1 * (-4), 1, 1],
     [1, 1, 1 * (-4), 1, 1]]
d = [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [-4, -4,-4,-4,-4],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]]

image = cv2.imread('b512.jpg', 0)
iw = np.zeros((image.shape[0], image.shape[1]))

identity = cv2.filter2D(src=image, ddepth=-1, kernel=np.array(c))

identity = m.pixel_log255(identity)

identity = identity.flatten()

for i, val in enumerate(identity):
    if val > 50:
        identity[i] = 255
    else:
        identity[i] = 0

grad = np.reshape(identity, (image.shape[0], image.shape[1]))
im = Image.fromarray(grad)
if im.mode != 'RGB':
    im = im.convert('RGB')
im.save("cc.jpg")
#
# image_aa = cv2.imread('aa.jpg', 0)
# image_bb = cv2.imread('bb.jpg', 0)
# image_cc = cv2.imread('cc.jpg', 0)
# image_dd = cv2.imread('dd.jpg', 0)
#
# iw_all = np.zeros((image_aa.shape[0], image_aa.shape[1]))
#
# for i in range(image_aa.shape[0]):
#     for j in range(image_aa.shape[1]):
#         numbers = [image_aa[i][j], image_bb[i][j], image_cc[i][j], image_dd[i][j]]
#         iw_all[i][j] = max(numbers)
#
# im = Image.fromarray(iw_all)
# if im.mode != 'RGB':
#     im = im.convert('RGB')
# im.save("max.jpg")
