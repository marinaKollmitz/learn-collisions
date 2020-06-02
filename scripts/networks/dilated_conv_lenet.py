import torch.nn as nn
import torch.nn.functional as F


class DilatedFCNLeNet(nn.Module):
    def __init__(self, im_size):
        super(DilatedFCNLeNet, self).__init__()

        # net properties
        self.net_size = im_size
        self.downsample_factor = 1

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5, dilation=2)

        # calculate size of last filter, based on input shape
        last_filter_size = int(im_size / 4) - 3
        if im_size % 4 != 0:
            print("DilatedFCNLeNet Warning: net input should be dividable by 4!")

        self.fc1 = nn.Conv2d(16, 120, last_filter_size, dilation=4)

        # sparse connect fc1 and fc2 to keep number of parameters similar to original LeNet
        self.fc2 = nn.Conv2d(120, 21, 2, dilation=2)
        self.fc3 = nn.Conv2d(21, 2, 2)

    def forward(self, x):

        # convolutional part
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))

        # convolutionalized fully connected part
        f1 = F.relu(self.fc1(c2))
        f2 = F.relu(self.fc2(f1))
        out = self.fc3(f2)  # softmax is automatically added with CrossEntropyLoss

        return out
