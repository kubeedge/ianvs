import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class AOA_dataset(Dataset):
    # all_data = np.loadtxt(path, delimiter=',')
    #在 init 中载入数据，先读入数据
    def __init__(self,path):
        self.path = path
        all_data = np.loadtxt(path, delimiter=',')
        self.x_data = torch.from_numpy(all_data[:, :-1]).float().unsqueeze(1)
        self.y_data  = torch.from_numpy(all_data[:, -1]).float().unsqueeze(1)
        self.len = len(all_data)
    # 在 getitem 中返回相应位置的数据
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    #在 len 中返回整个数据的长度
    def __len__(self):
        return self.len
       


if __name__ == '__main__':
    path = '../data/train/1_train.txt'
    data = AOA_dataset(path)
    for i in iter(data):
        print(i)
    print(len(data))
    train_dataloader = DataLoader(data,batch_size=32,shuffle=True,drop_last=False,)