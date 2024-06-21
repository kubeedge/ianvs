from torch.utils.data import Dataset,DataLoader,random_split
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

# path = r'data/processed/1.txt'
# path = r'data/Scene Partition/1_rural/1.txt'
class AOA_dataset(Dataset):
    # all_data = np.loadtxt(path, delimiter=',')
    def __init__(self,filepath):
        all_data = np.loadtxt(path, delimiter=',')
        self.x_data = torch.from_numpy(all_data[:, 10:-1]).float().unsqueeze(1)

        self.y_data  = torch.from_numpy(all_data[:, 5]).float().unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return len(self.x_data)




class AoACNN(torch.nn.Module):
    def __init__(self, flat_num=13, output_dim=1):
        super(AoACNN, self).__init__()
        self.output_dim = output_dim
        self.flat_num = flat_num
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            # input [batchsize,1,312], output [batchsize,8,156]
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=3)  # input [batchsize,8,156], output [batchsize,8,52]
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            # input [batchsize,8,52], output [batchsize,16, 26]
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=3, stride=2) # input [batchsize,16,26], output [batchsize,16,13]
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            # input [batchsize,16,13], output [batchsize,32, 13]
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=3, stride=2) # input [batchsize,16,26], output [batchsize,16,13]
        )
        # self.conv4 = torch.nn.Sequential(
        #     torch.nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1), # input [batchsize,16,13], output [batchsize,32, 13]
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.ReLU(),
        # )
        if self.output_dim == 1:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(32 * self.flat_num, 256), torch.nn.ReLU(),
                # torch.nn.Dropout(drop_p),
                torch.nn.Linear(256, 200), torch.nn.ReLU(),
                torch.nn.Linear(200, self.output_dim)
            )
        else:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(32 * self.flat_num, 256), torch.nn.ReLU(),
                # torch.nn.Dropout(drop_p),
                torch.nn.Linear(256, 200), torch.nn.ReLU(),
                torch.nn.Linear(200, self.output_dim)
            )
        # self.dp = torch.nn.Dropout(drop_p)

    def forward(self, x):
        # x = x.cuda()
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        # x = self.conv4(x)

        x = self.fc(x.view(x.size(0), -1))
        # print(x.shape)
        return x


if __name__ == "__main__":

    path = r'data/Scene Partition/1_rural/1.txt'
    mydataset = AOA_dataset(path)

    train_loader = DataLoader(dataset=mydataset, batch_size=32, shuffle=True, num_workers=2)

    model = AoACNN()
    criterion = torch.nn.MSELoss()  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 优化器优化参
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1.准备数据
            inputs, labels = data   # Tensor type

            # 2.前向传播
            y_pred = model(inputs)
            print(labels.size())
            print(y_pred.size())
            loss = criterion(y_pred, labels)
            lss = torch.abs(y_pred-labels)
            r2 = torch.sum(lss)
            print(r2.item())
            print(epoch, i, loss.item())

            # 3.后向传播
            optimizer.zero_grad()
            loss.backward()

            # 4.更新
            optimizer.step()

    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data  # Tensor type
            y_pred = model(inputs)
            loss = abs(y_pred, labels)



# all_data = np.loadtxt(path, delimiter=',')
# all_features = all_data[:, 10:-1]
# all_labels = all_data[:, 5]
# print(all_data.shape)