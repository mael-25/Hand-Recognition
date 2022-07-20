import copy
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, dropout=0.01):
        """input size: 96x96"""
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # self.device = torch.device("cuda")

        # self.conv1 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.ReLU())  # 48x48

        # self.conv2 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.ReLU())  # 24x24

        # self.conv3 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.ReLU())  # 12x12

        # self.conv4 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.ReLU())  # 6x6

        # self.conv5 = nn.Sequential(nn.Conv2d(3, 32, (4, 4)),
        #                            nn.ReLU())  # 3x3

        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 5),  # 92x92
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))  # 46x46

        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3),  # 44x44
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))  # 22x22

        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 5),  # 18x18
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))  # 9x9

        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, 5),  # 5x5
                                   nn.ReLU(),
                                   nn.Dropout(dropout))

        self.conv5 = nn.Sequential(nn.Conv2d(64, 32, 5),  # 1x1
                                   nn.ReLU(),
                                   nn.Dropout(dropout))

        self.resizeTo = 32

        # self.conv6 = nn.Sequential(nn.Conv2d(16, 18, (4, 4)),
        #                            nn.ReLU())  # 1x1

        self.linear1 = nn.Linear(self.resizeTo, 18)

        # self.conv7 = nn.Sequential(nn.Conv2d(32, 18, (1,1)))
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, xb):
        # print(1)
        xb = self.conv1(xb)
        # xb = self.activation1(xb)
        # print(1)

        xb = self.conv2(xb)
        # xb = self.activation2(xb)

        # print(1)
        xb = self.conv3(xb)
        # xb = self.activation3(xb)
        xb = self.conv4(xb)

        xb = self.conv5(xb)
        # print(xb.shape)
        # xb = self.activation4(xb)

        xb = xb.reshape(-1, self.resizeTo)
        x = copy.copy(xb)
        xb = self.linear1(xb)
        xb = self.soft(xb)

        return xb, x
