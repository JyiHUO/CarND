import torch
from torch.autograd import Variable
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import fbeta_score
from .Tt import *

def resize_image(image, size):
    # resize an image to the given size
    return image.resize(size, Image.ANTIALIAS)

def sample(model, img):
    # pil image
    model.train(False)
    model.cuda()
    t = T.Compose([
        cv_resize(224, 224),
        T.ToTensor()
    ])
    img_t = t(img)
    img_v = Variable(img_t, volatile=True).cuda()
    img_v = img_v.unsqueeze(0)
    pred =  model(img_v)
    return pred.data.cpu().numpy()[0][0]