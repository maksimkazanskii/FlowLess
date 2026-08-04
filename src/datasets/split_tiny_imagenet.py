import os

from torch.utils.data import Subset
from torchvision import datasets, transforms


class SplitTinyImageNet:

    TASKS = [
        list(range(i, i + 40))
        for i in range(0, 200, 40)
    ]

    def __init__(
            self,
            root="./data/tiny-imagenet-200",
    ):

        transform_train = transforms.Compose([
            transforms.RandomCrop(
                64,
                padding=8,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4802, 0.4481, 0.3975),
                std=(0.2302, 0.2265, 0.2262),
            ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4802, 0.4481, 0.3975),
                std=(0.2302, 0.2265, 0.2262),
            ),
        ])

        train_root = os.path.join(root, "train")
        val_root = os.path.join(root, "val")

        self.train_dataset = datasets.ImageFolder(
            train_root,
            transform=transform_train,
        )

        self.test_dataset = datasets.ImageFolder(
            val_root,
            transform=transform_test,
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
                Subset(
                    self.train_dataset,
                    train_idx,
                )
            )

            self.test_tasks.append(
                Subset(
                    self.test_dataset,
                    test_idx,
                )
            )

    def num_tasks(self):

        return len(self.TASKS)

    def get_task(
            self,
            task_id,
    ):

        return (
            self.train_tasks[task_id],
            self.test_tasks[task_id],
        )

    def get_train_task(
            self,
            task_id,
    ):

        return self.train_tasks[task_id]

    def get_test_task(
            self,
            task_id,
    ):

        return self.test_tasks[task_id]

    def get_classes(
            self,
            task_id,
    ):

        return self.TASKS[task_id]


if __name__ == "__main__":

    dataset = SplitTinyImageNet()

    print(
        f"Number of tasks: {dataset.num_tasks()}"
    )

    for task_id in range(dataset.num_tasks()):

        train_task, test_task = dataset.get_task(
            task_id
        )

        print(
            f"Task {task_id + 1}"
        )

        print(
            f"Classes: {dataset.get_classes(task_id)}"
        )

        print(
            f"Train samples: {len(train_task)}"
        )

        print(
            f"Test samples: {len(test_task)}"
        )

        print()