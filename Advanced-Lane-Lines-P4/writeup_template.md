# Advanced Lane Finding Project

### Overview

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"




### Camera Calibration


The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

And after applying the function `undistort`, I got the result like this:

![](output_images/correct.jpg)

It looks almost the same. Right? 

Nope, you can discovery some different about the white car between two images. It seems that the white car in picture two became larger.

#### 2. Gradient Threshold and Color Threshold

I used a combination of color and gradient thresholds to generate a binary image 

- In order to extract the yellow lane line, I only using `S` channel with threshold (130,200)

![S_channel](output_images/Figure_2.png)

- And then I campute the gradient magnitude for gray image with threshold (50,200) and ksize 3

![gray_gard](output_images/Figure_3.png)

- At last, I combine them to achieve the whole thing

![all_staff](output_images/Figure_1.png)

#### 3. Perspective Transform

The code for my perspective transform includes a function called `warper()`, which appears in lines 107 .  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warp](output_images/Figure_4.png)

After doing the wrap the binary image. It looks like they are almost parallel.:

![warp](output_images/figure_6.png)



#### 4. Fit the line to calculate curvature

Firstly, I cut the sky and plot histogram for perspective transform image , in order to find which part from the image containing lane line with huge probability:

![histogram](output_images/histogram.png)

Then I use the slide windows to search for the lane line and the procedure can be devide into servel steps:
- Finding the midpoint for the binary image
- use the static result from histogram to find maximun pixel
- use slide window to calculate the mean point, in order to move forward
- use the point what we find to fit the polynominal line

![find_line](output_images/find_line.png)

The result is amazing good. Then I reuse the parameter I fit, so that I can easily track the line in frame to framevedio:

![reuse](output_images/slide_win_curve.png)

I can't wait to apply it into the real picture:


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function `cal_curvature` in `all_staff.py`line 332 describes how to use it. It looks likes this:
```python
cal_curvature(ploty, left_fit_cr, right_fit_cr)
```
Firstly, I accept the parameter which is fitted from perspective transform image. And then I do some Partial derivative according to Radius of Curvature [formula](https://www.intmath.com/applications-differentiation/8-radius-curvature.php):
$$$

f(y)=Ay^2 +By+C
$$$
$$$
f′(y)=\frac{dx}{dy}=2Ay+B
$$$

$$$
f′′(y)=\frac{d^2x}{dy^2}=2A
$$$

$$$
R_{curve} = \frac{(1 + (2Ay + B)^2)^{(\frac{3}{2})}}{|2A|}
$$$

### Help!! I can not know how to calculate the vehicle with respect to center



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![p](output_images/result.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](res_video/project_video_res.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Bright to dark is so strange and unstable. I hope that reviewer can give some advice
- I can not know how to calculate the vehicle with respect to center. I hope that reviewer can give some advice
-  My curvature somethimes can get very large. Is this strange?
-  Dotted line make the region unstable. How can I do smoothing?
