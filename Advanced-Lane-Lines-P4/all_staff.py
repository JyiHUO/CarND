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
import pickle
import os

chessboard_dir = 'camera_cal/calibration*.jpg'
test_image = 'test_images/test4.jpg' # should read in RGB
obj_img_dir = 'obj_img_point.pkl'

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


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.ones_like(img)*100

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def binarize(img, s_thresh=(120, 255), sx_thresh=(20, 255), l_thresh=(40, 255)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    # h_channel = hls[:,:,0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    # sobelx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255))
    # l_channel_col=np.dstack((l_channel,l_channel, l_channel))
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    channels = 255 * np.dstack((l_binary, sxbinary, s_binary)).astype('uint8')
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary == 1))] = 1
    # binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
    return binary, channels

def binary_color(img, sobel_kernel=3, gray_threshold=(60,200), S_thresold=(100,200), L_threshold=(205,256)):
    '''
    :param img: from undistort_img
    :param sobel_kernel:
    :param gray_threshold:
    :param color_thresold:
    :return: gray image to warpped
    '''

    # LAB
    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = LAB[:, :, 0]
    L_binary = np.zeros_like(L)
    L_binary[(L > L_threshold[0]) & (L < L_threshold[1])] = 1


    # color thresold
    # S channel
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = HLS[:, :, 2]
    S_binary = np.zeros_like(S)
    S_binary[(S > S_thresold[0]) & (S < S_thresold[1])] = 1
    # S_binary = grad(S_binary, y=False, sobel_kernel=7)

    # L channel
    hL = HLS[:, :, 1]
    hL_binary = np.zeros_like(L)
    hL_binary[(hL > 200) & (hL < 255)] = 1

    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_binary = grad(gray)

    binary = np.zeros_like(L)
    binary[(L_binary == 1) | (S_binary == 1) ] = 1
    return S_binary

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def dir_threshold(gray, sobel_kernel=3, thresh=(np.pi/3, np.pi/2)):
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobel_y,abs_sobel_x)
    direction = np.absolute(direction)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mask

def grad(one_chanel,sobel_kernel=3, x = True, y = True, thresold=(0,255)):
    sobely = np.zeros_like(one_chanel)
    sobelx = np.zeros_like(one_chanel)
    if y:
        sobely = cv2.Sobel(one_chanel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    if x:
        sobelx = cv2.Sobel(one_chanel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # grad
    gradmag = np.sqrt(sobely**2 + sobelx**2) # cal magnitiue
    gradmag = (gradmag/np.max(gradmag)*255).astype(np.uint8) # convert to uint8

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag > thresold[0]) & (gradmag < thresold[1])] = 1
    return binary_output

# def binary_color(img, sobel_kernel=3, gray_threshold=(50,200), S_thresold=(100,255), L_threshold=(215,256)):
#     '''
#     :param img: from undistort_img
#     :param sobel_kernel:
#     :param gray_threshold:
#     :param color_thresold:
#     :return: gray image to warpped
#     '''
#
#
#     ## grad threshold
#
#     # direction grad
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray_direction = dir_threshold(gray, thresh=(np.pi/6, np.pi/2))
#
#     # sobelx
#     sobelx = grad(gray, sobel_kernel=3, y=False, thresold=(20,200))
#
#     ## color thresold
#     # S channel
#     HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     S = HLS[:, :, 2]
#     S_binary = np.zeros_like(S)
#     S_binary[(S > S_thresold[0]) & (S < S_thresold[1])] = 1
#
#     # grad S_binary
#     S_grad = grad(S_binary, y=False)
#
#     # R channel
#     LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     L = LAB[:, :, 0]
#     L_binary = np.zeros_like(L)
#     L_binary[(L > L_threshold[0]) & (L < L_threshold[1])] = 1
#
#     binary = np.zeros_like(S_binary)
#     # binary[(L_binary == 1) | (S_binary == 1) ] = 1
#
#     # dstack them and smoothing
#     # print (np.dstack([sobelx, S_binary, S_grad, L_binary]).shape)
#     binary_m = np.mean(np.dstack([ S_binary, L_binary*2]),axis=2)
#     binary = grad(binary_m, y=False)
#     return binary

def mask_S_channel(img):
    src1 = np.int64(
        [[[510,480],
          [180,710],
          [370,710],
          [690,480]]]
    )
    src2 = np.int64(
        [[[610,480],
          [1060,710],
          [1200,710],
          [790,480]]]
    )
    mask1 = region_of_interest(img, src1)
    mask2 = region_of_interest(img, src2)
    mask = np.zeros_like(mask1)
    mask[(mask1==1) | (mask2==1)] = 1
    return mask

def mask_grad(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.int64(
        [[[(img_size[0] / 2) - 70, img_size[1] / 2 + 100],
          [((img_size[0] / 6) -100), img_size[1]],
          [(img_size[0] * 5 / 6) + 250, img_size[1]],
          [(img_size[0] / 2 + 150), img_size[1] / 2 + 100]]])

    mask_img = region_of_interest(img, src)

    # plt.imshow(mask_img, cmap='gray')
    # plt.show()
    return mask_img

def warp_M(img):
    '''
    :param img: can be binary image returen for colour and gradient threshold
    :return: binary warped
    '''
    img_size = (img.shape[1], img.shape[0])

    bottom_left = [220, 720]
    bottom_right = [1110, 720]
    top_left = [570, 470]
    top_right = [722, 470]
    src = np.float32([[top_left,bottom_left,bottom_right,top_right]])

    bottom_left = [320, 720]
    bottom_right = [920, 720]
    top_left = [320, 1]
    top_right = [920, 1]
    dst = np.float32([[top_left,bottom_left,bottom_right,top_right]])

    Minv = cv2.getPerspectiveTransform(dst, src)

    M = cv2.getPerspectiveTransform(src, dst)
    wrapper = cv2.warpPerspective(img, M, img_size)
    return wrapper, Minv


def slide_windows(binary_warped, nwindows=9, margin=100, minpix=5):
    '''
    :param binary_warped: from perspective transform
    :param nwindows: num of slide wins
    :param margin: size of wins
    :param minpix: pixel in windows to run
    :return: left, right and y position to draw
    '''
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)  # cut the sky
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(binary_warped.shape[0] / nwindows)

    nonzero = binary_warped.nonzero()
    nonzero_row = nonzero[0]
    nonzero_col = nonzero[1]

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_left = leftx_current - margin
        win_xleft_right = leftx_current + margin

        win_xright_left = rightx_current - margin
        win_xright_right = rightx_current + margin

        # cv2.rectangle(out_img, (win_xleft_left, win_y_low), (win_xleft_right, win_y_high), (0, 0, 255), 2)
        # cv2.rectangle(out_img, (win_xright_left, win_y_low), (win_xright_right, win_y_high), (0, 0, 255), 2)

        good_left_ids = ((nonzero_row >= win_y_low) & (nonzero_row <= win_y_high) &\
                         (nonzero_col >= win_xleft_left) & (nonzero_col <= win_xleft_right)).nonzero()[0]

        good_right_ids = ((nonzero_row >= win_y_low) & (nonzero_row >= win_y_high) &\
                          (nonzero_col >= win_xright_left) & (nonzero_col <= win_xright_right)).nonzero()[0]

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

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return ploty, left_fitx, right_fitx, left_fit, right_fit

def acc_frame_to_frame(binary_warped, left_fit, right_fit, margin = 100):
    '''
    :param binary_warped: from perspective transform
    :param left_fit: from slide windows
    :param right_fit:
    :param margin:
    :return: left, right and y position to draw
    '''
    nonzero = binary_warped.nonzero()
    nonzero_row = nonzero[0]
    nonzero_col = nonzero[1]

    left_lane_inds = (
    (nonzero_col > (left_fit[0] * nonzero_row ** 2 + left_fit[1] * nonzero_row + left_fit[2] - margin)) & \
    (nonzero_col < (left_fit[0] * nonzero_row ** 2 + left_fit[1] * nonzero_row + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzero_col > (right_fit[0] * nonzero_row ** 2 + right_fit[1] * nonzero_row + right_fit[2] - margin)) & \
    (nonzero_col < (right_fit[0] * nonzero_row ** 2 + right_fit[1] * nonzero_row + right_fit[2] + margin)))

    leftx = nonzero_col[left_lane_inds]
    lefty = nonzero_row[left_lane_inds]
    rightx = nonzero_col[right_lane_inds]
    righty = nonzero_row[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return ploty, left_fitx, right_fitx, left_fit, right_fit


def img_region(warped, left_fitx, right_fitx, ploty, Minv, undist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def img_region_no_T(warped, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    original_warp = np.dstack((warped, warped, warped))*255

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    result = cv2.addWeighted(original_warp, 1, color_warp, 0.5, 0)

    return result

def cal_curvature(ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # fit new polynomials
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

def cal_center_offset(left_fitx, right_fitx):
    xm_per_pix = 3.7 / 700
    center = (left_fitx[719] + right_fitx[719])/2
    return abs(640 - center)*xm_per_pix

def dstack_img(img, r=False, g=False, b=False):
    if r:
        return np.dstack([img*255, img, img])
    elif g:
        return np.dstack([img, img* 255, img])
    elif b:
        return np.dstack([img, img, img*255])
    else:
        return np.dstack([img, img, img])*255

def pipline(img, objpoints,imgpoints):
    undist = undistort_img(img, objpoints, imgpoints)

    # normal
    # binary_img = binary_color(undist)
    binary_img, _ = binarize(undist)
    # img_read_show(img, binary_img, gray=True)
    mask_img = mask_grad(binary_img)
    # img_read_show(binary_img, mask_img, gray=True)
    binary_warped, Minv = warp_M(mask_img)
    img_read_show(undist, binary_warped, gray=True)
    # abnormal
    warped_color, Minv = warp_M(undist)
    binary_warped_color = binary_color(warped_color)

    ploty, left_fitx, right_fitx, left_fit, right_fit = slide_windows(binary_warped, nwindows=9, margin=20, minpix=5)
    # ploty, left_fitx, right_fitx, left_fit, right_fit = acc_frame_to_frame(binary_warped, left_fit, right_fit, margin = 20)
    # print (left_fitx)
    result = img_region(binary_warped, left_fitx, right_fitx, ploty, Minv, undist)
    result_warp = img_region_no_T(binary_warped, left_fitx, right_fitx, ploty)

    # put text
    curvature = cal_curvature(ploty, left_fitx, right_fitx)
    center_offset = cal_center_offset(left_fitx, right_fitx)
    c_s = "Curvature: left:{:.2f}km  right:{:.2f}km".format(curvature[0]/1000, curvature[1]/1000)
    c_o = "Center offset: {:.2f}m".format(center_offset)

    # concat many image
    result = cv2.resize(result, (640, 320))
    binary_img = dstack_img(cv2.resize(binary_img, (640, 320)), b=True)
    # mask_img = dstack_img(cv2.resize(mask_img, (640, 320)), r=True)
    warped_color = cv2.resize(warped_color, (320, 320))
    binary_warped_color = dstack_img(cv2.resize(binary_warped_color, (320,320)), r=True)
    result_warp = cv2.resize(result_warp, (640, 320))

    mask_img = np.concatenate([warped_color, binary_warped_color], axis=1)
    print (mask_img.shape)
    temp1 = np.concatenate([result, result_warp], axis=0)
    temp2 = np.concatenate([binary_img, mask_img], axis=0)
    temp = np.concatenate([temp1, temp2], axis=1)

    cv2.putText(temp, c_s, (22,22), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
    cv2.putText(temp, c_o, (22,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
    return temp

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


## For the slide windows test
# binary_warped = Image.imread('output_images/img_perspective_transform.jpg')[:, :, 0] # shape(720, 1280, 4) -> (720, 1280)
# histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0) # cut the sky
#
# out_img = np.dstack([binary_warped, binary_warped, binary_warped])*255
#
# midpoint = np.int(histogram.shape[0]/2)
# leftx_base = np.argmax(histogram[:midpoint])
# rightx_base = np.argmax(histogram[midpoint:])+midpoint
#
# nwindows = 9
#
# window_height = np.int(binary_warped.shape[0]/nwindows)
#
# nonzero = binary_warped.nonzero()
# nonzero_row = nonzero[0]
# nonzero_col = nonzero[1]
#
# leftx_current = leftx_base
# rightx_current = rightx_base
#
# margin = 100
#
# minpix = 50
#
# left_lane_inds = []
# right_lane_inds = []
#
# for window in range(nwindows):
#     win_y_low = binary_warped.shape[0] - (window+1) * window_height
#     win_y_high = binary_warped.shape[0] - window*window_height
#
#     win_xleft_left = leftx_current - margin
#     win_xleft_right = leftx_current + margin
#
#     win_xright_left = rightx_current - margin
#     win_xright_right = rightx_current + margin
#
#     cv2.rectangle(out_img, (win_xleft_left, win_y_low), (win_xleft_right,win_y_high), (0,0,255), 2)
#     cv2.rectangle(out_img, (win_xright_left, win_y_low), (win_xright_right, win_y_high), (0,0,255),2)
#
#     good_left_ids = ((nonzero_row >= win_y_low)&(nonzero_row <= win_y_high) &\
#                      (nonzero_col>=win_xleft_left) & (nonzero_col<=win_xleft_right)).nonzero()[0]
#
#     good_right_ids = ((nonzero_row>=win_y_low)&(nonzero_row>=win_y_high) &\
#                       (nonzero_col>=win_xright_left) & (nonzero_col<=win_xright_right)).nonzero()[0]
#
#     left_lane_inds.append(good_left_ids)
#     right_lane_inds.append(good_right_ids)
#
#     if len(good_left_ids) > minpix:
#         leftx_current = np.int(np.mean(nonzero_col[good_left_ids]))
#     if len(good_right_ids) > minpix:
#         rightx_current = np.int(np.mean(nonzero_col[good_right_ids]))
#
# # concatenate the arrays
# left_lane_inds = np.concatenate(left_lane_inds)
# right_lane_inds = np.concatenate(right_lane_inds)
#
# leftx = nonzero_col[left_lane_inds]
# lefty = nonzero_row[left_lane_inds]
# rightx = nonzero_col[right_lane_inds]
# righty = nonzero_row[right_lane_inds]
#
# # Fit the line according to y
# left_fit = np.polyfit(lefty, leftx, 2)
# right_fit = np.polyfit(righty, rightx, 2)

## Visualization
# ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#
# out_img[nonzero_row[left_lane_inds], nonzero_col[left_lane_inds]] = [255,0,0]
# out_img[nonzero_row[right_lane_inds], nonzero_col[right_lane_inds]] = [0,255,0]
#
# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.show()


## skip the slide windows when you know where the line are
# nonzero = binary_warped.nonzero()
# nonzero_row = nonzero[0]
# nonzero_col = nonzero[1]
#
# margin = 100
# left_lane_inds = ((nonzero_col > (left_fit[0]*nonzero_row**2 + left_fit[1]*nonzero_row + left_fit[2]-margin))&\
#                   (nonzero_col < (left_fit[0]*nonzero_row**2 + left_fit[1]*nonzero_row + left_fit[2]+margin)))
# right_lane_inds = ((nonzero_col > (right_fit[0]*nonzero_row**2 + right_fit[1]*nonzero_row + right_fit[2]-margin))&\
#                    (nonzero_col < (right_fit[0]*nonzero_row**2 + right_fit[1]*nonzero_row + right_fit[2]+margin)))
#
# leftx = nonzero_col[left_lane_inds]
# lefty = nonzero_row[left_lane_inds]
# rightx = nonzero_col[right_lane_inds]
# righty = nonzero_row[right_lane_inds]
#
# left_fit = np.polyfit(lefty, leftx, 2)
# right_fit = np.polyfit(righty, rightx, 2)
#
# ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Create an image to draw on and an image to show the selection window
# out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# window_img = np.zeros_like(out_img)
# # Color in left and right line pixels
# out_img[nonzero_row[left_lane_inds], nonzero_col[left_lane_inds]] = [255, 0, 0]
# out_img[nonzero_row[right_lane_inds], nonzero_col[right_lane_inds]] = [0, 0, 255]
#
# # Generate a polygon to illustrate the search window area
# # And recast the x and y points into usable format for cv2.fillPoly()
# left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
# left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
#                               ploty])))])
# left_line_pts = np.hstack((left_line_window1, left_line_window2))
# right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
# right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
#                               ploty])))])
# right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
# # Draw the lane onto the warped blank image
# cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
# cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
# result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
# plt.imshow(result)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()

## test pipline
# objpoints, imgpoints = r_obj_img_point(chessboard_dir)
# print(objpoints)
# with open('obj_img_point.pkl', 'wb') as f:
#     pickle.dump([objpoints, imgpoints], f)

# with open('obj_img_point.pkl', 'rb') as f:
#     objpoints, imgpoints = pickle.load(f)
#
# img = Image.imread(test_image)
# undist = undistort_img(img, objpoints, imgpoints)
# binary_img = binary_color(undist, sobel_kernel=3, gray_threshold=(45,255), color_thresold=(110,255))
# binary_warped, Minv = warp_M(binary_img)
# ploty, left_fitx, right_fitx, left_fit, right_fit = slide_windows(binary_warped, nwindows=9, margin=100, minpix=50)
# ploty, left_fitx, right_fitx = acc_frame_to_frame(binary_warped, left_fit, right_fit, margin = 100)
# result = img_region(binary_warped, left_fitx, right_fitx, ploty, Minv, undist)
#
# plt.imshow(result)
# plt.show()

## test a list of img
with open(obj_img_dir, 'rb') as f:
    objpoints, imgpoints = pickle.load(f)

# img  = Image.imread('test_images/'+'straight_lines1.jpg')
# res = pipline(img, objpoints, imgpoints)
# plt.imshow(res)
# plt.show()

for img_name in os.listdir('test_images/'):
    img = Image.imread('test_images/' +img_name)
    print ('test_images/' +img_name)
    res = pipline(img, objpoints, imgpoints)
    plt.imshow(res)
    plt.show()



