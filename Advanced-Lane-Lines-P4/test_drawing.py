import cv2
import matplotlib.image as Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.imread('test_images/test5.jpg')

wrap_zero = np.zeros_like(image).astype(np.uint8)
# color_wrap = np.dstack((wrap_zero, wrap_zero, wrap_zero))
# print (color_wrap.shape)

leftx = np.ones(200) * 100
lefty = np.arange(200)
pts_left = np.array([np.transpose(np.vstack([leftx, lefty]))])
print (pts_left.shape)

rightx = np.ones(200) * 200
righty = np.arange(200)+100
pts_right = np.array([np.transpose(np.vstack([rightx, righty]))]) # flip?
print (pts_right[0, 0])
pts_right[0, 0] = [400,100]
print (pts_right.shape)

pts = np.hstack([pts_left, pts_right])
# print (pts.dtype)
# print (np.int_(pts))
cv2.fillPoly(wrap_zero, np.int_(pts), (0,255,0))

plt.imshow(wrap_zero)
plt.show()