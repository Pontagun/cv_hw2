import cv2
import numpy as np
from PIL import Image
import math
import main as m

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
     [1 * (-4), 1 * (-4), 1 * (-4), 1 * (-4), 1 * (-4)],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]]

image = cv2.imread('kirsh.jpg', 0)
iw = np.zeros((image.shape[0], image.shape[1]))

identity = cv2.filter2D(src=image, ddepth=-1, kernel=np.array(d))

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
im.save("d.jpg")
