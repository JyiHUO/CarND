from mode_DenseNetl import *
from T import Gray, Resize
import torchvision.transforms as transform
import torch
from PIL import Image
import os
from torch.autograd import Variable
from prepare_data import *
from data_loader import get_loader

def sample(image_dir):
    model = transfer_DenseNet()
    model.load_state_dict(torch.load('model/model.pkl'))
    model.eval()

    transforms = transform.Compose([
        Resize((32,32)),
        transform.ToTensor(),
        Gray()
    ])

    images = os.listdir(image_dir)
    res = []

    for img_name in images:
        img = Image.open(os.path.join(image_dir,img_name))
        t_val = Variable(transforms(img).unsqueeze(0), volatile=True)
        output = model(t_val)
        sort_t = torch.sort(output.data.squeeze())
        res.append((img, sort_t))

    return res

def sample_test(X_test, y_test, use_cuda):
    model = transfer_DenseNet()
    model.load_state_dict(torch.load('model/model.pkl'))
    model.eval()
    model.cuda()

    test_dataloader = get_loader(X_test, y_test, batch_size=128, shuffle=False)
    all_equal = 0.0
    all_count = X_test.shape[0]

    for i, (input, target) in enumerate(test_dataloader):
        model.train(False)
        if use_cuda:
            input = Variable(input, volatile=True).cuda()
            target = Variable(target.squeeze(), volatile=True).cuda()
        else:
            input = Variable(input, volatile=True)
            target = Variable(target.squeeze(), volatile=True)

        output = model(input)


        _, pred = torch.max(output.data, 1)
        all_equal += torch.sum(pred == target.data)
    print('-' * 20)
    print('test:\n')
    print( all_equal / all_count)
    print('-' * 20)

if __name__ == "__main__":
    data = prepare_data('data/train.p', 'data/valid.p', 'data/test.p')
    sample_test(data[4], data[5], True)