import torch
from torch import Tensor
from torch.nn import Module


# X = torch.tensor([[1.,2.,3.],[4.,4.,5.],[6.,8.,8.],[9.,5.,7.]])
# p0 = 0.4
# p1 = 0.6
# p05 = 1- p0 - p1
# training = True

def dropout(X, p0, p1, training, inplace=False):
    p05 = max(1 - p0 - p1, 0)
    # print(X.sum())
    weights = torch.Tensor([p0, p05, p1])   # 采样权重
    mask = torch.multinomial(weights, len(X.view(-1)), replacement=True)
    mask = mask.view(X.shape)/2
    # print((mask * X / (0*p0 + 0.5*p05 + 1*p1)).sum())
    return mask * X / (0*p0 + 0.5*p05 + 1*p1)

# dropout(X, p0, p1, training)


class _DropoutNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p1: float = 0.2, p2: float = 0.2, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        # if p < 0 or p > 1:
        #     raise ValueError("dropout probability has to be between 0 and 1, "
        #                      "but got {}".format(p))
        self.p1 = p1
        self.p2 = p2
        self.inplace = inplace

    # def extra_repr(self) -> str:
    #     return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return dropout(input, self.p1, self.p2, self.training, self.inplace)

if __name__ == 'main':
    do = Dropout(0.2, 0.4)
    input_ = torch.tensor([[1.,2.,3.],[4.,4.,5.],[6.,8.,8.],[9.,5.,7.]])
    do(input_)