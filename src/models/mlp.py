import torch.nn as nn


class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 64)

        self.relu = nn.ReLU()

        self.head = nn.Linear(64, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        h1 = self.relu(self.l1(x))
        h2 = self.relu(self.l2(h1))
        h3 = self.relu(self.l3(h2))

        logits = self.head(h3)

        return logits

    def features(self, x):

        x = x.view(x.size(0), -1)

        h1 = self.relu(self.l1(x))
        h2 = self.relu(self.l2(h1))
        h3 = self.relu(self.l3(h2))

        return {
            "layer1": h1,
            "layer2": h2,
            "layer3": h3
        }