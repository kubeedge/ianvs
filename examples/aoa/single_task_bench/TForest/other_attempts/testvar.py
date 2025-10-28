import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tree import AoACNN
# plt.style.use(['science', 'ieee'])
logging.basicConfig(level=logging.INFO)
from tree import TForest_Tree,AOA_dataset,Tree
import os
import statsmodels.api as sm
import scienceplots
from tqdm import tqdm
from torch.utils.data import DataLoader
# plt.style.use(['science', 'ieee'])

import torch
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
device  = torch.device('cuda')


def test_main():
    # 载入4大模型和分配器
    # tree_final = joblib.load('final_tree.joblib')
    # fcp_tree = joblib.load('fcp_tree.joblib')
    # mtp_tree = joblib.load('mtp_tree.joblib')
    # stp_tree = joblib.load('stp_tree.joblib')
    # selector = joblib.load('selector.joblib')
    tree_initial =  joblib.load('1_tree.joblib')
    path = r'data/verification/1_ver.txt'
    test_dataset = AOA_dataset(path)
    pre_all = []

    y_labels = test_dataset.y_data.numpy().reshape(1, -1)
    for i in test_dataset.x_data:
        feature = i.numpy().reshape(312, )
        inputs = torch.unsqueeze(i, dim=1).to(device)
        model = AoACNN()
        model.load_state_dict(torch.load('model.pth'))
        model = model.to(device)
        with torch.no_grad():

            pre = model(inputs)
            pre = pre.item()
            pre_all.append(pre)
    print(pre_all-y_labels)



    # with torch.no_grad():
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=2)
    #     # y_labels = test_dataset.y_data.numpy().reshape(1, -1)
    #     # for i in test_dataset.x_data:
    #         # feature = i.numpy().reshape(312, )
    #         # inputs = torch.unsqueeze(i, dim=1).to(device)
    #
    #     # model = tree_initial[0].model.to(device)
    #     model = AoACNN()
    #     model.load_state_dict(torch.load('model.pth'))
    #     model = model.to(device)
    #     for i, data in enumerate(test_loader, 0):
    #         inputs, labels = data  # Tensor type
    #
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         y_pred = model(inputs)
    #         pre_loss = (y_pred-labels).cpu().numpy()
    #         print(pre_loss)
    #         print(i)



if __name__ == '__main__':
    # tforest_main()
    # test里面开始，构建test的数据集
    # task_selector()
    test_main()
    print(1)