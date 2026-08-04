import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:

    def __init__(
            self,
            model,
            device,
            lambda_ewc=1000.0
    ):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc

        self.fisher = {}
        self.star_params = {}

    @torch.no_grad()
    def store_params(self):

        self.star_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def compute_fisher(
            self,
            dataset,
            batch_size=128
    ):

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        criterion = nn.CrossEntropyLoss()

        self.model.eval()

        for x, y in loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.model.zero_grad()

            logits = self.model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            for n, p in self.model.named_parameters():

                if p.grad is None:
                    continue

                fisher[n] += (
                        p.grad.detach() ** 2
                )

        for n in fisher:
            fisher[n] /= len(dataset)

        if len(self.fisher) == 0:
            self.fisher = fisher
        else:
            for n in fisher:
                self.fisher[n] += fisher[n]

    def penalty(self):

        if len(self.fisher) == 0:
            return torch.tensor(
                0.0,
                device=self.device
            )

        loss = 0.0

        for n, p in self.model.named_parameters():

            if n not in self.fisher:
                continue

            loss += (
                    self.fisher[n]
                    * (p - self.star_params[n]) ** 2
            ).sum()

        return self.lambda_ewc * loss