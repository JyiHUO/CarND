import numpy as np
from PIL import ImageChops, Image
# Image preprocessing and data augmentation
class Gray(object):
    def __call__(self, tensor):
        _, H, W = tensor.size()
        R = tensor[0]
        G = tensor[1]
        B = tensor[2]
        tensor = 0.299*R + 0.587*G + 0.114*B
        tensor = tensor.view(1, H, W)
        return tensor

class Rotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, pil):
        if np.random.random() >= 0.9:
            angle = np.random.randint(self.angle)
            return pil.rotate(angle)
        elif np.random.random() <= 0.1:
            angle = np.random.randint(self.angle)
            return pil.rotate(-angle)
        else:
            return pil

class Shift(object):
    def __init__(self, size_prob):
        self.size_prob = size_prob

    def __call__(self, img):
        w, h = img.size
        w_ = int(w*self.size_prob)
        h_ = int(h*self.size_prob)
        if np.random.random() > 0.1:
            return img
        return self.ImgOfffSet(img, w_, h_)

    def ImgOfffSet(self, Img, xoff, yoff):
        width, height = Img.size
        c = ImageChops.offset(Img, xoff, yoff)
        c.paste((0, 0, 0), (0, 0, xoff, height))
        c.paste((0, 0, 0), (0, 0, width, yoff))
        return c

class Zoom(object):
    def __init__(self, zoom_prob):
        self.zoom_prob = zoom_prob / 2

    def __call__(self, img):
        w, h = img.size
        top_left = self.zoom_prob * w
        bottom_right = w - self.zoom_prob * w
        if np.random.random() > 0.1:
            return img
        return img.transform((w, h), Image.EXTENT, (top_left, top_left, bottom_right, bottom_right))

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize(self.size, Image.ANTIALIAS)