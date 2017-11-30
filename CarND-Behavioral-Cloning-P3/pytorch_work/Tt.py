from torchvision import transforms
import cv2
import numpy as np

class cv_crop(object):
    def __init__(self, top_t=80, bottom_t=180):
        self.top_t = top_t
        self.bottom_t = bottom_t

    def __call__(self, img):
        return img[self.top_t:self.bottom_t, :, :]

class cv_sobel(object):
    def __init__(self, kernel=3):
        self.kernel=kernel

    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel)

        gradmap = np.sqrt(sobelx**2 + sobely**2)

        gradmap = (255.*gradmap/np.max(gradmap)).astype(np.uint8)
        return gradmap.reshape(gradmap.shape+(1,))

class cv_resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        return cv2.resize(img, (self.height, self.width))