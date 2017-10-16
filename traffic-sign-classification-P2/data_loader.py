import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from T import *
import torchvision.transforms as transform
# customer dataset
class myDataset(Dataset):
    def __init__(self, X, y, transform_train=None):
        # initialize some data
        self.X = X
        self.y = y
        self.transform = transform_train
        self.transform_val = transform.Compose([
            transform.ToTensor(),
            Gray()
        ])

    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        target = self.y[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.transform_val(img)

        return img, torch.LongTensor([int(target)])

    def __len__(self):
        return self.X.shape[0]

def get_loader(X, y, transforms=None, batch_size=256, shuffle=True, num_workers=4):
    dataset = myDataset(X, y, transforms)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader