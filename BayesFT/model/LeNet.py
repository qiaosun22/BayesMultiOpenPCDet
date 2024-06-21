import torch.nn as nn
import torch.nn.functional as F
from model.randomnoise import GaussLayer, LaplaceLayer
import torch
from torch import nn
from scipy.stats import beta
import numpy as np


'''以下ERM版'''
# def dropout_layer(X, dropout_rate, training):    
#     return X #mask * X * (X.sum() / (mask * X).sum())


'''以下加Beta分布噪声版'''
def dropout_layer(X, dropout_rate, training):

    a = dropout_rate
    b = 1 - dropout_rate
    
    mask = torch.tensor(beta.rvs(a, b, size=tuple(X.shape)).astype(np.float32))
    mask = mask.to(torch.device("cuda"))  # .float()
    
    return mask * X * (X.sum() / (mask * X).sum())


'''以下加上正态分布噪声版'''
# def dropout_layer(X, dropout_rate, training):
#     '''
#     dropout rate是概率分布中的唯一可变参数，根据这个参数所决定的分布生成随机噪声，并加载到输入中，得到输出
#     '''
#     # assert 0 <= dropout_rate <= 1
#     # if training == False:
#     #     return X
#     # if dropout_rate == 1:
#     #     return torch.zeros_like(X)
#     # if dropout_rate == 0:
#     #     return X
    
#     mean = 0.  # dropout_rate#1.
#     std = dropout_rate
    
#     mask = torch.normal(mean, std, tuple(X.shape), out=None) #→ Tensor #std和mean都是标量，此时需要size指定shape
#     # mask = (torch.rand(X.shape) > dropout_rate).float()  # 1 if True, else 0
    
#     mask = mask.to(torch.device("cuda"))
#     dropout_rate = (torch.tensor(dropout_rate)).to(torch.device("cuda"))
#     X = X.to(torch.device("cuda"))
#     return mask + X # / (1.0 - dropout_rate)


'''以下dropout原版'''
# def dropout_layer(X, dropout_rate, training):
#     assert 0 <= dropout_rate <= 1
#     if training == False:
#         return X
#     if dropout_rate == 1:
#         return torch.zeros_like(X)
#     if dropout_rate == 0:
#         return X
    
#     mask = (torch.rand(X.shape) > dropout_rate).float()  # 1 if True, else 0
    
#     mask = mask.to(torch.device("cuda"))
#     dropout_rate = (torch.tensor(dropout_rate)).to(torch.device("cuda"))
#     # X = X.to(torch.device("cuda"))
#     return mask * X / (1.0 - dropout_rate)



class LeNet_Bayes(nn.Module):
    def __init__(self, p0, p1, p2, p3, p4):
        super().__init__()
        
        self.training = False if self.eval==True else True
        
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        
        self.conv1 = nn.Conv2d(1, 6, 5, 1)  # 28 by 28
        self.do1 = dropout_layer  # self.do1 = nn.Dropout2d(p=p0)
        
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.do2 = dropout_layer
        
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.do3 = dropout_layer
        
        self.fc1 = nn.Linear(120, 84)
        self.do4 = dropout_layer
        
        self.fc2 = nn.Linear(84, 10)
        self.do5 = dropout_layer

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 24 by 24
        out = self.do1(out, dropout_rate=self.p0, training=self.training)
        out = F.max_pool2d(out, 2)  # 12 by 12

        out = F.relu(self.conv2(out))  # 8 by 8
        out = self.do2(out, dropout_rate=self.p1, training=self.training)
        out = F.max_pool2d(out, 2)  # 4 by 4

        out = F.relu(self.conv3(out))  # 1 by 1
        out = out.view(out.size(0), -1)
        out = self.do3(out, dropout_rate=self.p2, training=self.training)

        out = F.relu(self.fc1(out))
        out = self.do4(out, dropout_rate=self.p3, training=self.training)
        
        out = self.fc2(out)
        out = self.do5(out, dropout_rate=self.p4, training=self.training)
        return out



class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)  # 28 by 28
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 24 by 24

        out = F.max_pool2d(out, 2)  # 12 by 12

        out = F.relu(self.conv2(out))  # 8 by 8

        out = F.max_pool2d(out, 2)  # 4 by 4

        out = F.relu(self.conv3(out))  # 1 by 1
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out



class LeNet_Gauss(nn.Module):
    def __init__(self, p0, p1, p2, p3, p4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)  # 28 by 28
        self.do1 = GaussLayer(p0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.do2 = GaussLayer(p1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.do3 = GaussLayer(p2)
        self.fc1 = nn.Linear(120, 84)
        self.do4 = GaussLayer(p3)
        self.fc2 = nn.Linear(84, 10)
        self.do5 = GaussLayer(p4)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 24 by 24
        out = self.do1(out)
        out = F.max_pool2d(out, 2)  # 12 by 12

        out = F.relu(self.conv2(out))  # 8 by 8
        out = self.do2(out)
        out = F.max_pool2d(out, 2)  # 4 by 4

        out = F.relu(self.conv3(out))  # 1 by 1
        out = out.view(out.size(0), -1)
        out = self.do3(out)

        out = F.relu(self.fc1(out))
        out = self.do4(out)
        out = self.fc2(out)
        out = self.do5(out)
        return out


class LeNet_Laplace(nn.Module):
    def __init__(self, p0, p1, p2, p3, p4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)  # 28 by 28
        self.do1 = LaplaceLayer(p0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.do2 = LaplaceLayer(p1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.do3 = LaplaceLayer(p2)
        self.fc1 = nn.Linear(120, 84)
        self.do4 = LaplaceLayer(p3)
        self.fc2 = nn.Linear(84, 10)
        self.do5 = LaplaceLayer(p4)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 24 by 24
        out = self.do1(out)
        out = F.max_pool2d(out, 2)  # 12 by 12

        out = F.relu(self.conv2(out))  # 8 by 8
        out = self.do2(out)
        out = F.max_pool2d(out, 2)  # 4 by 4

        out = F.relu(self.conv3(out))  # 1 by 1
        out = out.view(out.size(0), -1)
        out = self.do3(out)

        out = F.relu(self.fc1(out))
        out = self.do4(out)
        out = self.fc2(out)
        out = self.do5(out)
        return out


class LeNet_Best(nn.Module):
    def __init__(self, p0=0.767587903356465, p1=0.7177749741580626, p2=0.7271342310396831, p3=0.7655704645227732,
                 p4=0.216092236164781):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)  # 28 by 28
        self.do1 = nn.Dropout2d(p=p0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.do2 = nn.Dropout2d(p=p1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.do3 = nn.Dropout(p=p2)
        self.fc1 = nn.Linear(120, 84)
        self.do4 = nn.Dropout(p=p3)
        self.fc2 = nn.Linear(84, 10)
        self.do5 = nn.Dropout(p=p4)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 24 by 24
        out = self.do1(out)
        out = F.max_pool2d(out, 2)  # 12 by 12

        out = F.relu(self.conv2(out))  # 8 by 8
        out = self.do2(out)
        out = F.max_pool2d(out, 2)  # 4 by 4

        out = F.relu(self.conv3(out))  # 1 by 1
        out = out.view(out.size(0), -1)
        out = self.do3(out)

        out = F.relu(self.fc1(out))
        out = self.do4(out)
        out = self.fc2(out)
        out = self.do5(out)
        return out


class LeNet_12(nn.Module):
    def __init__(self, p0=0.1, p1=0.1, p2=0.1, p3=0.1, p4=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)  # 28 by 28
        self.do1 = nn.Dropout2d(p=p0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.do2 = nn.Dropout2d(p=p1)
        self.conv3 = nn.Conv2d(16, 120, 4, 1)
        self.do3 = nn.Dropout(p=p2)
        self.fc1 = nn.Linear(120, 84)
        self.do4 = nn.Dropout(p=p3)
        self.fc2 = nn.Linear(84, 10)
        self.do5 = nn.Dropout(p=p4)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 24 by 24
        out = self.do1(out)
        out = F.max_pool2d(out, 2)  # 12 by 12

        out = F.relu(self.conv2(out))  # 8 by 8
        out = self.do2(out)
        out = F.max_pool2d(out, 2)  # 4 by 4

        out = F.relu(self.conv3(out))  # 1 by 1
        out = out.view(out.size(0), -1)
        out = self.do3(out)

        out = F.relu(self.fc1(out))
        out = self.do4(out)
        out = self.fc2(out)
        out = self.do5(out)
        return out
