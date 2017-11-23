import cv2
import numpy as np
from skimage.feature import hog


def hog_extract(img, img_c="YUV", orientations=9, pix_per_cell=8, cell_per_block=2, ravel=False, concat=False):
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

    if concat:
        return np.concatenate(res)
    else:
        return res

def bin_spatial(img, size=32):
    size = (size, size)
    c1 = cv2.resize(img[:, :, 0], size).ravel()
    c2 = cv2.resize(img[:, :, 1], size).ravel()
    c3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack([c1, c2, c3])


def color_extract(img, nbins=32):
    c1 = np.histogram(img[:, :, 0], bins=nbins)
    c2 = np.histogram(img[:, :, 1], bins=nbins)
    c3 = np.histogram(img[:, :, 2], bins=nbins)

    return np.concatenate([c1[0], c2[0], c3[0]])

def find_cars(img, clf, ystart=360, ystop=-96,orientations=9, train_img_size=64, pix_per_cell=8, cell_per_block=2, predict_threshold=0.5):
    '''
    :param img: (720x1280)
    :param clf: concatence 3c hog, color, bin spatial
    :param ystart:
    :param ystop:
    :param orientations:
    :param scale:
    :param train_img_size: image size when training
    :param pix_per_cell:
    :param cell_per_block:
    :param spatial_size:
    :param hist_bins:
    :return:
    '''

    img = img[ystart:ystop, :, :] # (264, 1280, 3)
    hog_array1, hog_array2, hog_array3 = hog_extract(img, orientations=orientations, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    # (32, 159, 2, 2, 9)

    # do some calculation for times
    train_hog_large = int((train_img_size - pix_per_cell) / pix_per_cell) # (7,7,2,2,9) = 7
    hog_h = hog_array1.shape[0] # 32
    hog_w = hog_array1.shape[1] # 159
    hog_step_h = int(hog_h - train_hog_large + 1)
    hog_step_w = int(hog_w - train_hog_large + 1)

    # print(hog_step_w)
    # print (hog_step_h)

    bbox_temp = []
    bbox_list = []
    X = []
    # do slide windows
    for h in range(int(hog_step_h)):
        for w in range(int(hog_step_w)):
            # bbox
            check_top_left = (w*pix_per_cell, h*pix_per_cell+ystart)
            check_bottom_right = (w*pix_per_cell+train_img_size, h*pix_per_cell+train_img_size+ystart)

            # extract feature
            img_sub = img[h*pix_per_cell:h*pix_per_cell+train_img_size, w*pix_per_cell:w*pix_per_cell+train_img_size, :]
            # print (img_sub.shape)
            bin_f = bin_spatial(img_sub)
            # print (bin_f.shape)
            color_f = color_extract(img_sub)
            # print(color_f.shape)
            hog_f = np.concatenate([hog_array1[h:h+train_hog_large, w:w+train_hog_large].ravel(),
                                   hog_array2[h:h+train_hog_large, w:w+train_hog_large].ravel(),
                                   hog_array3[h:h+train_hog_large, w:w+train_hog_large].ravel()])
            # print(hog_f.shape)
            features = np.concatenate([hog_f, color_f, bin_f])

            # predict
            # print (features.shape)
            X.append(features)

            bbox_temp.append([check_top_left, check_bottom_right])
    X = np.array(X)
    y_pred = clf.predict_proba(X)[:, 1]
    # print (y_pred.shape)
    for i,v in enumerate(y_pred):
        if v > predict_threshold:
            bbox_list.append(bbox_temp[i])
    return bbox_list

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold=0):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    for label in range(1, labels[1]+1):
        nonzeros = (labels[0] == label).nonzero() # (row, col)
        top_left = (nonzeros[1].min(), nonzeros[0].min())
        bottom_right = (nonzeros[1].max(), nonzeros[0].max())
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    return img

