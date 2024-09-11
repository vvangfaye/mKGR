from abc import ABC, abstractmethod
from typing import Tuple
from utils.hyperbolic import hyp_distance_multi_c
import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class F2(Regularizer):
    def __init__(self, weight: float):
        super(F2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]
    
class hyperbolic_distance(Regularizer):
    def __init__(self, weight: float):
        super(hyperbolic_distance, self).__init__()
        self.weight = weight

    def forward(self, factors):
        dist = 0
        for i in range(len(factors)):
            for j in range(i+1,len(factors)):
                dist += self.weight * torch.sum(hyp_distance_multi_c(factors[i],factors[j],c=1))
        return dist / factors[0].shape[0]
                
class Euluc(Regularizer):
    def __init__(self, weight: float):
        super(Euluc, self).__init__()
        self.weight = weight

    def forward(self, factors):
        dist = 0
        for i in range(len(factors)):
            for j in range(i+1,len(factors)):
                dist += self.weight * torch.sum((factors[i]-factors[j])**2)
                
        return dist / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]
