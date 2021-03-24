import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('data\cat_puke.tiff')     # รูป
mask = cv2.imread('data\mask.tiff',0)      # ลายขีดเขียน

dst_TELEA = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
dst_NS = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

plt.subplot(221), plt.imshow(img)
plt.title('damaged image')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.title('mask')
plt.subplot(223), plt.imshow(dst_TELEA)
plt.title('TELEA')
plt.subplot(224), plt.imshow(dst_NS)
plt.title('NS')

fig = plt.gcf()
fig.canvas.set_window_title('Image Inpainting Test')
plt.tight_layout()
plt.show()