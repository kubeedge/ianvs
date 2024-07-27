import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_dim=316, hidden_num=4, output_dim=316):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, hidden_num), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_num, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, input_seqs):
        encoded = self(input_seqs)
        decoded = self.decoder(encoded)
        return decoded


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


class DCNN(nn.Module):
    def __init__(self, seq_len, flat_num=120, input=None, output=None):  # seq_len=1 输入的segment的长度，input维度是312
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 1))
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 40, (5, 1))
        self.bn2 = nn.BatchNorm2d(40)
        if seq_len <= 20:
            self.conv3 = nn.Conv2d(40, 20, (2, 1))
        else:
            self.conv3 = nn.Conv2d(40, 20, (3, 1))
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d((2, 1))
        self.lin1 = nn.Linear(input * flat_num, 400)
        self.lin2 = nn.Linear(400, output)
        self.lin3 = nn.Linear(21600, 10)

    def forward(self, input_seqs):
        h = input_seqs.unsqueeze(1)
        h = F.relu(F.tanh(self.conv1(h)))
        h = self.pool(h)
        h = F.relu(F.tanh(self.conv2(h)))
        h = self.pool(h)
        h = F.relu(F.tanh(self.conv3(h)))
        h = h.view(h.size(0), h.size(1), h.size(2) * h.size(3))
        # print(h.shape)
        h = self.lin1(h)
        h = F.relu(F.tanh(torch.sum(h, dim=1)))
        h = self.normalize(h[:, :, None, None])
        h = self.lin2(h[:, :, 0, 0])
        return h

    def normalize(self, x, k=1, alpha=2e-4, beta=0.75):
        # x = x.view(x.size(0), x.size(1) // 5, 5, x.size(2), x.size(3))#
        # y = x.clone()
        # for s in range(x.size(0)):
        #     for j in range(x.size(1)):
        #         for i in range(5):
        #             norm = alpha * torch.sum(torch.square(y[s, j, i, :, :])) + k
        #             norm = torch.pow(norm, -beta)
        #             x[s, j, i, :, :] = y[s, j, i, :, :] * norm
        # x = x.view(x.size(0), x.size(1) * 5, x.size(3), x.size(4))
        return x

