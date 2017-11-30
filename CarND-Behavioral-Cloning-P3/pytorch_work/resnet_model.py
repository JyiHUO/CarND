import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class resnet_(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18()
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear1 = nn.Linear(resnet.fc.in_features, 512)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(128,32)
        self.linear4 = nn.Linear(32, 1)

    def forward(self, img):
        x = self.resnet(img)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return x