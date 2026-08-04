from abc import ABC, abstractmethod


class ContinualMethod(ABC):

    @abstractmethod
    def before_task(self, model):
        pass

    @abstractmethod
    def loss(self, model, ce_loss):
        pass

    @abstractmethod
    def after_task(self, model):
        pass