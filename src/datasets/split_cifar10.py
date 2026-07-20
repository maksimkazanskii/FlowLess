import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


class SplitCIFAR10:

    TASKS = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]

    def __init__(self, root="./data"):

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        self.train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=train_transform,
        )

        self.test_dataset = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=test_transform,
        )

        self.train_tasks = []
        self.test_tasks = []

        self._build_tasks()

    def _build_tasks(self):

        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets

        for classes in self.TASKS:

            train_idx = [
                i
                for i, y in enumerate(train_targets)
                if y in classes
            ]

            test_idx = [
                i
                for i, y in enumerate(test_targets)
                if y in classes
            ]

            self.train_tasks.append(
                Subset(self.train_dataset, train_idx)
            )

            self.test_tasks.append(
                Subset(self.test_dataset, test_idx)
            )

    def num_tasks(self):
        return len(self.TASKS)

    def get_task(self, task_id):
        return (
            self.train_tasks[task_id],
            self.test_tasks[task_id],
        )

    def get_classes(self, task_id):
        return self.TASKS[task_id]