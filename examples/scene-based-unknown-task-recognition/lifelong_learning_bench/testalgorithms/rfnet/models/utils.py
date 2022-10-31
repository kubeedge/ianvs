import os
import shutil
import time
import pprint
import torch
import numpy as np
import torch.nn as nn
#  --- functional helper ---
def category_mean(data, label, label_max):
    '''compute mean for each category'''
    one_hot_label = one_hot(label, label_max)
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    return torch.mm(data.view(1, -1), one_hot_label).squeeze(0)

def category_mean1(data, label, label_max):
    '''compute mean for each category
       only return centers for given categories'''
    labelset = torch.unique(label, sorted=True)
    one_hot_label = one_hot(label, label_max)
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    output = torch.mm(data.view(1, -1), one_hot_label).squeeze(0)
    return output[labelset]

def category_mean2(data, label, label_max):
    '''compute mean for each category, based on a matrix'''
    one_hot_label = one_hot(label, label_max)
    data = torch.gather(data, 1, label.unsqueeze(1))
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    return torch.mm(data.view(1, -1), one_hot_label).squeeze(0)

def category_mean3(data, label, label_max):
    '''compute mean for each category, each row corresponds to an elements'''
    one_hot_label = one_hot(label, label_max)
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    return torch.mm(one_hot_label.t(), data)

def category_mean4(data, label, label_max):
    '''compute mean for each category, each row corresponds to an elements
       only return centers for given categories'''
    labelset = torch.unique(label, sorted=True)
    one_hot_label = one_hot(label, label_max)
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    pre_center = torch.mm(one_hot_label.t(), data)
    return pre_center[labelset, :]

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()    
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
                shutil.rmtree(path)
                os.mkdir(path)
    else:
        os.mkdir(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
