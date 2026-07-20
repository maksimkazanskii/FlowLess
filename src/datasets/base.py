from abc import ABC, abstractmethod


class ContinualDataset(ABC):

    @abstractmethod
    def num_tasks(self):
        pass

    @abstractmethod
    def get_task(self, task_id):
        pass