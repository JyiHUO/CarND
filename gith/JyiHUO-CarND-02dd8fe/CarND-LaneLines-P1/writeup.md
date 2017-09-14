# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipline is very simple, here is my code：
```python
def pipline_image(img, kernel_size=5, low_threshold=90, high_threshold=100, rho=1, theta=np.pi/180, threshold=50, min_line_len=200,max_line_gap=150):
    img_gray = grayscale(img)
    img_blur = gaussian_blur(img_gray, kernel_size)
    img_canny = canny(img_blur, low_threshold, high_threshold)
    vertices = np.array( [[[400,325],[580, 325],[850, 530],[150, 530]]], dtype=np.int32 )
    img_region = region_of_interest(img_canny, vertices)
    img_hough = hough_lines(img_region, rho, theta, threshold, min_line_len, max_line_gap)
    img_weig = weighted_img(img_hough, img, α=0.8, β=1., λ=0.)
    return img_weig
```

My pipeline consisted of 5 steps. 
- I converted the images to grayscale
- Blured the image with `kernel_size = 5`
- Used `canny` algorithm to find the edge of picture
- Changed the value of vertices so the shape of the region looks like a Trapezoid
- Within this region, I use `hough_lines` function to find a series of points
- I didn't modify `draw_line()`function, so I use it to draw a straight line directly
- Finally, I added the line into the original picture.




### 2. Identify potential shortcomings with your current pipeline


Help!!! I can not use it into the curve lin. Can you give some advice?


### 3. Suggest possible improvements to your pipeline

Maybe I should pay more attention to modify `draw_line` function.
