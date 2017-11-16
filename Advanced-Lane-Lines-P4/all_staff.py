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

out_img = np.dstack([binary_warped, binary_warped, binary_warped])*255

midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:])+midpoint

nwindows = 9

window_height = np.int(binary_warped.shape[0]/nwindows)

nonzero = binary_warped.nonzero()
nonzero_row = nonzero[0]
nonzero_col = nonzero[1]

leftx_current = leftx_base
rightx_current = rightx_base

margin = 100

minpix = 50

left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    win_y_low = binary_warped.shape[0] - (window+1) * window_height
    win_y_high = binary_warped.shape[0] - window*window_height

    win_xleft_left = leftx_current - margin
    win_xleft_right = leftx_current + margin

    win_xright_left = rightx_current - margin
    win_xright_right = rightx_current + margin

    cv2.rectangle(out_img, (win_xleft_left, win_y_low), (win_xleft_right,win_y_high), (0,0,255), 2)
    cv2.rectangle(out_img, (win_xright_left, win_y_low), (win_xright_right, win_y_high), (0,0,255),2)

    good_left_ids = ((nonzero_row >= win_y_low)&(nonzero_row <= win_y_high) &\
                     (nonzero_col>=win_xleft_left) & (nonzero_col<=win_xleft_right)).nonzero()[0]

    good_right_ids = ((nonzero_row>=win_y_low)&(nonzero_row>=win_y_high) &\
                      (nonzero_col>=win_xright_left) & (nonzero_col<=win_xright_right)).nonzero()[0]

    left_lane_inds.append(good_left_ids)
    right_lane_inds.append(good_right_ids)

    if len(good_left_ids) > minpix:
        leftx_current = np.int(np.mean(nonzero_col[good_left_ids]))
    if len(good_right_ids) > minpix:
        rightx_current = np.int(np.mean(nonzero_col[good_right_ids]))

# concatenate the arrays
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

leftx = nonzero_col[left_lane_inds]
lefty = nonzero_row[left_lane_inds]
rightx = nonzero_col[right_lane_inds]
righty = nonzero_row[right_lane_inds]

# Fit the line according to y
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

## Visualization
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzero_row[left_lane_inds], nonzero_col[left_lane_inds]] = [255,0,0]
out_img[nonzero_row[right_lane_inds], nonzero_col[right_lane_inds]] = [0,255,0]

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.show()

# plt.imshow(out_img)
# plt.show()