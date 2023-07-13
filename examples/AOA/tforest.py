import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee'])
logging.basicConfig(level=logging.INFO)
from tree import TForest_Tree, AOA_dataset, Tree
import os
import statsmodels.api as sm
import scienceplots
# from tqdm import tqdm
from torch.utils.data import DataLoader
# plt.style.use(['science', 'ieee'])

import torch
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from tree import AoACNN

device = torch.device('cuda')


# 画图函数
def cdf_draw_each(data, label, color='#8ea3c2', marker='*', linestyle='-', markersize=3):
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(min(data), max(data))
    y = ecdf(x)
    # plt.step(x, y, label=label, color=color, markersize=markersize, marker=marker, linestyle=linestyle)
    plt.step(x, y, label=label)


# 测试误差函数
def test_main(tf_distribution):
    # 载入4大模型和分配器
    # tree_final = joblib.load('tf_final.joblib')
    tree_final = joblib.load(r'/gemini/code/tf_final.joblib')
    # fcp_tree = joblib.load('fcp_tree.joblib')
    # mtp_tree = joblib.load('mtp_tree.joblib')
    # stp_tree = joblib.load('stp_tree.joblib')
    # selector = joblib.load('selector.joblib')

    # 载入6个数据集，对应四个场景
    # path = r'data/test/'
    path = r'/gemini/data-2/data/test/'
    test_list = ['rural-1.txt', 'rural-2.txt', 'parkinglot-1.txt', 'highway-1.txt', 'surface-1.txt', 'surface-2.txt']
    for index in range(len(test_list)):
        # 6个测试场景
        test_dataset = AOA_dataset(path + test_list[index])
        # test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=2)

        TF_error = []

        # 求四个task编号  TF_task,MTP_task,FCP_task,RTP_task，STP不需要求
        y_labels = test_dataset.y_data.numpy().reshape(-1, )
        # 1
        for test_data in test_dataset.x_data:
            # 获取模型的序号,测试数据每行与模型对应的数据集每行做欧氏距离，求欧氏距离平均值，得到最小的欧氏距离对应的模型的序号
            # 得到模型序号从调用模型做预测，存储误差
            TF_dist = []
            for tf_index in range(len(tf_distribution)):
                a = tf_distribution[tf_index] - test_data
                # 一个训练集数据与测试集一行做差，求第二范式
                tf_list = torch.linalg.norm(tf_distribution[tf_index] - test_data, axis=1)  # [len(data),312]
                tf_dist = torch.mean(tf_list)
                # mtp_dist= np.mean(np.linalg.norm(mtp_feature,axis = 1))
                TF_dist.append(tf_dist)
            # 得到这一行数据最好的模型序号
            TF_task = TF_dist.index(min(TF_dist))
            print(tree_final[TF_task].model_name)
            model = AoACNN()
            # model_path = '/gemini/data-1/model/mtp/'
            # 载入对应模型
            model_name = tree_final[TF_task].model_name
            model.load_state_dict(torch.load(model_name))
            model = model.to(device)
            # 处理输入数据
            inputs = torch.unsqueeze(test_data, dim=1).to(device)
            # 预测
            with torch.no_grad():
                pre = model(inputs)
                pre = pre.item()
                TF_error.append(pre)
        # 2
        TF_error = abs(np.array(TF_error) - y_labels)
        print(f'{test_list[index]}的TF_error为:{TF_error}')
        # print(len(TF_error))
        # 改
        save_path = r'/gemini/output/'
        np.savetxt(save_path + test_list[index], TF_error, delimiter=',', fmt='%.8f')

    # return TF_error

if __name__ == '__main__':
    # tree_list = joblib.load('tf_final.joblib')
    tree_list = joblib.load(r'/gemini/code/tf_final.joblib')
    tf_distribution = []
    for i in range(len(tree_list)):
        loader = DataLoader(dataset=tree_list[i].train_dataset, batch_size=100000, shuffle=False)
        for data in loader:
            inputs, labels = data
            # 降维 [len(data),1,312]->[len(data),312]
            # inputs = inputs.mean(axis = 0)
            inputs = torch.squeeze(inputs)
            tf_distribution.append(inputs)
    len(tf_distribution)
    print(tf_distribution[0])
    print(tf_distribution[0].shape)
    test_main(tf_distribution)