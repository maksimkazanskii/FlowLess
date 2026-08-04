from .split_mnist import SplitMNIST
from .split_fashion_mnist import SplitFashionMNIST
from .split_cifar10 import SplitCIFAR10
from .split_tiny_imagenet import SplitTinyImageNet

class Dataset:
    def __init__(self, name):
        name = name.lower()
        if name == "mnist":
            self.dataset = SplitMNIST()
        elif name in ["fashion", "fashion_mnist"]:
            self.dataset = SplitFashionMNIST()
        elif name in ["cifar", "cifar10"]:
            self.dataset = SplitCIFAR10()
        elif name in {"tinyimagenet", "tiny_imagenet"}:
            self.dataset = SplitTinyImageNet()
        else:
            raise ValueError(name)

    def num_tasks(self):
        return self.dataset.num_tasks()

    def get_task(self, task_id):
        return self.dataset.get_task(task_id)