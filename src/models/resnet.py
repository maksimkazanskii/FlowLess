import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class ResNet18(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        backbone = resnet18(weights=None)

        # Remove the original classifier
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool

        self.fc = nn.Linear(512, num_classes)

    def features(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)

        f1 = torch.flatten(F.adaptive_avg_pool2d(z1, 1), 1)
        f2 = torch.flatten(F.adaptive_avg_pool2d(z2, 1), 1)
        f3 = torch.flatten(F.adaptive_avg_pool2d(z3, 1), 1)
        f4 = torch.flatten(F.adaptive_avg_pool2d(z4, 1), 1)

        return {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
        }

    def forward(self, x):

        z = self.features(x)["layer4"]
        return self.fc(z)