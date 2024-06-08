import torch
from torch import nn
from torch.nn import functional as F


# based on https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1 (visited on May 22, 2022)
class VGG16(nn.Module):
    def __init__(self, n_classes, **kwargs):
        super(VGG16, self).__init__()

        self.n_classes = n_classes

        n_filters = 16
        self.conv1_1 = nn.Conv2d(
            in_channels=9, out_channels=n_filters, kernel_size=3, padding=1
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1
        )

        self.conv2_1 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=n_filters * 2,
            out_channels=n_filters * 2,
            kernel_size=3,
            padding=1,
        )

        self.conv3_1 = nn.Conv2d(
            in_channels=n_filters * 2,
            out_channels=n_filters * 4,
            kernel_size=3,
            padding=1,
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=n_filters * 4,
            out_channels=n_filters * 4,
            kernel_size=3,
            padding=1,
        )
        self.conv3_3 = nn.Conv2d(
            in_channels=n_filters * 4,
            out_channels=n_filters * 4,
            kernel_size=3,
            padding=1,
        )

        self.conv4_1 = nn.Conv2d(
            in_channels=n_filters * 4,
            out_channels=n_filters * 8,
            kernel_size=3,
            padding=1,
        )
        self.conv4_2 = nn.Conv2d(
            in_channels=n_filters * 8,
            out_channels=n_filters * 8,
            kernel_size=3,
            padding=1,
        )
        self.conv4_3 = nn.Conv2d(
            in_channels=n_filters * 8,
            out_channels=n_filters * 8,
            kernel_size=3,
            padding=1,
        )

        self.conv5_1 = nn.Conv2d(
            in_channels=n_filters * 8,
            out_channels=n_filters * 8,
            kernel_size=3,
            padding=1,
        )
        self.conv5_2 = nn.Conv2d(
            in_channels=n_filters * 8,
            out_channels=n_filters * 8,
            kernel_size=3,
            padding=1,
        )
        self.conv5_3 = nn.Conv2d(
            in_channels=n_filters * 8,
            out_channels=n_filters * 8,
            kernel_size=3,
            padding=1,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(65536, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.dropout(F.relu(self.conv2_2(x)), 0.3)
        x = self.maxpool(x)
        x = F.dropout(x, 0.3)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.dropout(F.relu(self.conv3_3(x)), 0.3)
        x = self.maxpool(x)
        # x = F.dropout(x,0.3)
        # x = F.relu(self.conv4_1(x))
        # x = F.relu(self.conv4_2(x))
        # x = F.dropout(F.relu(self.conv4_3(x)),0.3)
        # x = self.maxpool(x)
        # x = F.dropout(x,0.3)
        # x = F.relu(self.conv5_1(x))
        # x = F.relu(self.conv5_2(x))
        # x = F.dropout(F.relu(self.conv5_3(x)),0.3)
        # x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)  # dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x
