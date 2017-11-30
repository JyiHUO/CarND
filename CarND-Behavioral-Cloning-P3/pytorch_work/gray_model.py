import torch.nn as nn
import torch
import torch.nn.functional as F

class gray_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=(5,5), stride=2)
        self.N1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=(5,5), stride=2)
        self.N2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=(5,5), stride=2)
        self.N3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,3))
        self.N4 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(1536, 100)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(100,50)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(50,10)
        self.dropout3 = nn.Dropout()
        self.linear4 = nn.Linear(10,1)

    def forward(self, img):
        x = self.conv1(img)
        x = self.N1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.N2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.N3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.N4(x)
        x = F.relu(x)
        x = x.view(-1, 1536)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return x