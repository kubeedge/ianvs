import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(420)
x = torch.rand((500,20),dytpe = torch.float32)
y = torch.randint(low = 0,high = 3,size =(500,),dytpe = torch.float32)
