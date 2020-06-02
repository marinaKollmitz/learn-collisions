import torch.nn as nn
import torch.nn.functional as F


class FCNLeNet(nn.Module):
    def __init__(self, im_size):
        super(FCNLeNet, self).__init__()

        # net properties
        self.net_size = im_size
        self.downsample_factor = 4

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # calculate size of last filter, based on input shape
        last_filter_size = int(im_size / 4) - 3
        if im_size % 4 != 0:
            print("FCNLeNet Warning: net input should be dividable by 4!")

        self.fc1 = nn.Conv2d(16, 120, last_filter_size)
        self.fc2 = nn.Conv2d(120, 84, 1)
        self.fc3 = nn.Conv2d(84, 2, 1)

    def forward(self, x):
        # convolutional part
        c1 = nn.functional.relu(self.conv1(x))
        p1 = self.pool(c1)
        c2 = nn.functional.relu(self.conv2(p1))
        p2 = self.pool(c2)

        # convolutionalized fully connected part
        f1 = F.relu(self.fc1(p2))
        f2 = F.relu(self.fc2(f1))
        out = self.fc3(f2)  # softmax is automatically added with CrossEntropyLoss

        return out
