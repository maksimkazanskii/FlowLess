import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


class SplitFashionMNIST:

    TASKS = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]

    def __init__(self, root="./data"):

        transform = transforms.ToTensor()

        self.train_dataset = datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )

        self.test_dataset = datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform,
        )

        self.train_tasks = []
        self.test_tasks = []

        self._build_tasks()

    def _build_tasks(self):

        for classes in self.TASKS:

            train_idx = [
                i
                for i, (_, y) in enumerate(self.train_dataset)
                if y in classes
            ]

            test_idx = [
                i
                for i, (_, y) in enumerate(self.test_dataset)
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