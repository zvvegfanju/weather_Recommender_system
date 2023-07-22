import pandas as pd
import time
from utils import fix_seed_torch, draw_loss_pic
import argparse
from model import GCN
from Logger import Logger
from mydataset import MyDataset
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import sys

import torch
import torch_geometric


b = torch.eye(3)  # 定义一个单位矩阵
print(b)
#
# i = torch.LongTensor([[0, 1, 1],
#                      [2, 1, 0]])
# d = torch.tensor([3, 6, 9], dtype=torch.float)
# a = torch.sparse.FloatTensor(i, d, torch.Size([2, 3]))
# print(a)
# print(a.to_dense())
# print(a.add(a).to_dense())  # a + a
# b = torch.eye(3)  # 定义一个单位矩阵
# y = torch.sparse.mm(a, b)
# print(y)