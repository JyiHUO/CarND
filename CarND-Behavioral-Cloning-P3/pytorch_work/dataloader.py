from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T

class myDataset(Dataset):
    def __init__(self, image_dir, csv_dir, transform=None, center=True):
        self.image_dir = image_dir
        csv_file = pd.read_csv(csv_dir, header=None)
        if not center:
            left_arr = csv_file[[1,3]].values
            left_arr[:, 1] += 0.1
            center_arr = csv_file[[0,3]].values
            right_arr = csv_file[[2,3]].values
            right_arr[:, 1] -= 0.1
            self.data = np.concatenate([left_arr, center_arr, right_arr], axis=0)
        else:
            self.data = csv_file[[0,3]].values

        if transform == None:
            self.transform = T.ToTensor()
        else:
            self.transform = transform


    def __getitem__(self, index):
        temp = self.data[index]
        img_name = temp[0].split('/')[-1]
        img = cv2.imread(os.path.join(self.image_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = float(temp[1])
        return self.transform(img), torch.FloatTensor([target])


    def __len__(self):
        return len(self.data)

def get_loader(image_dir, csv_dir, batch_size=128,num_workers=4, transform=None, center=3):
    dataset = myDataset(image_dir, csv_dir, transform, center)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    return data_loader