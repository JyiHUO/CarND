import cv2
import numpy as np
from skimage.feature import hog
def hog_extract(img, img_c="YUV", orientations=9, pix_per_cell=8, cell_per_block=2, ravel=False):
    def hog_img(img_):
        features = hog(img_, orientations=orientations,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=False, feature_vector=ravel,
                       block_norm="L2-Hys")
        return features

    img_temp = None
    if img_c == "RGB":
        img_temp = img
    elif img_c == "HSV":
        img_temp = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif img_c == "HLS":
        img_temp = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif img_c == "YUV":
        img_temp = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif img_c == "LUV":
        img_temp = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif img_c == "YCrCb":
        img_temp = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    res = []
    for i in range(3):
        res.append(hog_img(img_temp[:, :, i]))

    return np.concatenate(res)

