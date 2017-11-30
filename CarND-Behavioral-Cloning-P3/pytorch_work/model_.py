import torch.nn as nn
import torch.nn.functional as F

class myModle(nn.Module):
    def __init__(self):
        super().__init__()
        #
        self.conv1 = nn.Conv2d(3, 24, stride=2, kernel_size=5)
        self.conv2 = nn.Conv2d(24, 36, stride=2, kernel_size=5)
        self.conv3 = nn.Conv2d(36, 48, stride=2, kernel_size=5)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.linear1 = nn.Linear(1152, 100)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.linear4 = nn.Linear(10, 1)



    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)

        return x
