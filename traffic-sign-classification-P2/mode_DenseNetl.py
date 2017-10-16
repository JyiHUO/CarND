import torch.nn as nn
import torchvision.models as models

class transfer_DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        denseNet = models.densenet121() # pretrained=True
        # self.denseNet = nn.Sequential(*list(denseNet.children())[:-1])
        self.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.denseNet = nn.Sequential(*list(denseNet.features.children())[1:])
        num_features = denseNet.classifier.in_features
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(num_features, 43)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, image):
        x = self.conv0(image)
        x = self.denseNet(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x