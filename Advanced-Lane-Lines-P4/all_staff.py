'''
- undistort camara
- binary image
- perspective transform
- color curve
'''
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as Image
import time

chessboard_dir = 'camera_cal/calibration*.jpg'
test_image = 'test_images/test4.jpg' # should read in RGB

# utils
def img_read_show(img_o, img_c, gray=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    f.tight_layout()
    ax1.imshow(img_o)
    if gray:
        ax2.imshow(img_c, cmap='gray')
    else:
        ax2.imshow(img_c)
    plt.show()


# Pipline
def r_obj_img_point(chess_board_dir):
    '''
    :param chess_board_dir: a list of image which is distorted or undistorted
    :return: the parameter which will be used by undistort_img
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(chess_board_dir)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints

def undistort_img(img, objpoints, imgpoints):
    """
    :param img_name: the image you want to undistort
    :param objpoints: from r_obj_img_point
    :param imgpoints: from r_obj_img_point
    :return:
    """

    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def binary_color(img, sobel_kernel=3, gray_threshold=(45,255), color_thresold=(110,255)):

    def grad(one_chanel):
        sobely = cv2.Sobel(one_chanel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobelx = cv2.Sobel(one_chanel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

        # grad
        gradmag = np.sqrt(sobely**2 + sobelx**2) # cal magnitiue
        gradmag = (gradmag/np.max(gradmag)*255).astype(np.uint8) # convert to uint8

        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag > gray_threshold[0]) & (gradmag < gray_threshold[1])] = 1
        return binary_output

    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_binary = grad(gray)

    # color thresold
    # S channel
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = HLS[:, :, 2]
    S_binary = np.zeros_like(S)
    S_binary[(S > color_thresold[0]) & (S < color_thresold[1])] = 1


    binary = np.zeros_like(gray)
    binary[(gray_binary == 1) | (S_binary == 1) ] = 1
    return binary

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) +40), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 70), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    # print(img.shape)
    # print(src)
    # print(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    wrapper = cv2.warpPerspective(img, M, img_size)
    return wrapper, M

## For camara distorted test
# s = time.time()
# o, i = r_obj_img_point(chessboard_dir)
# e = time.time()
# print ('points_time: ', e-s)
#
# test_img = 'test_images/test1.jpg'
# origin_img = cv2.imread(test_img)
# correct_img = undistort_img(test_img, o, i)
#
# ## cv2.imshow('origin', origin_img)
# cv2.imwrite('correct.jpg', correct_img)


## For binary image test
# img = Image.imread(test_image)
# img_read_show(img, binary_color(img), gray=True)

## For PerspectiveTransform test
# img = Image.imread(test_image)
# o, i = r_obj_img_point(chessboard_dir)
# img = undistort_img(img, o, i)
# img_read_show(img, warp(img)[0], False)

## For step 1 ~ 3
# img = Image.imread(test_image)
#
# o, i = r_obj_img_point(chessboard_dir)
# img = undistort_img(img, o, i)
#
# binary_img = binary_color(img)
#
# img_perspective_transform = warp(binary_img)[0]
# img_read_show(img, img_perspective_transform, True)

## Saving the perspective transform image
# img = Image.imread(test_image)
#
# o, i = r_obj_img_point(chessboard_dir)
# img = undistort_img(img, o, i)
#
# binary_img = binary_color(img)
#
# img_perspective_transform = warp(binary_img)[0]
#
# plt.imsave('img_perspective_transform.jpg',img_perspective_transform, cmap=plt.cm.gray)


## For the slide windows
binary_warped = Image.imread('output_images/img_perspective_transform.jpg')[:, :, 0] # shape(720, 1280, 4) -> (720, 1280)
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0) # cut the sky

out_img = np.dstack([binary_warped, binary_warped, binary_warped])*2

print (out_img.dtype)
print (np.sum(binary_warped))
print (np.sum(out_img))
plt.imshow(out_img)
plt.show()