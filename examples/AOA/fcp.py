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
from tree import AoACNN
import torch
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

device = torch.device('cuda')


def TForest_tree_initial():
    # 输入每个trace数据，然后训练一个tree存到列表
    tree_list = []
    # 11,12,21,33,34,42为测试集
    path = r'data/tforest/'
    # 后面再改成11啊
    for i in range(1, 11):
        model_path = r'model/tforest/'
        model_name = str(i)+'.pth'
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'verification/' + str(i) + '_ver.txt')
        tree = TForest_Tree(train_dataset, ver_dataset,model_path+model_name)
        tree.train()
        tree_list.append(tree)
    for i in range(13, 21):
        model_path = r'model/tforest/'
        model_name = str(i) + '.pth'
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'verification/' + str(i) + '_ver.txt')
        tree = TForest_Tree(train_dataset, ver_dataset,model_path+model_name)
        tree.train()
        tree_list.append(tree)
    for i in range(22, 33):
        model_path = r'model/tforest/'
        model_name = str(i) + '.pth'
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'verification/' + str(i) + '_ver.txt')
        tree = TForest_Tree(train_dataset, ver_dataset,model_path+model_name)
        tree.train()
        tree_list.append(tree)
    for i in range(35, 42):
        model_path = r'model/tforest/'
        model_name = str(i) + '.pth'
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'verification/' + str(i) + '_ver.txt')
        tree = TForest_Tree(train_dataset, ver_dataset,model_path+model_name)
        tree.train()
        tree_list.append(tree)

    return tree_list


# def tree_update(tree_list):
def check_fusion_type(score1, score2, tree1, tree2):
    # score1 = tree1.predict_score(tree_fused.model)
    # score2 = tree2.predict_score(tree_fused.model)

    if (score1 > tree1.r2) and (score2 > tree2.r2):
        return 0  # entire benefical
    elif score1 > tree1.r2:
        return 1  # self benefical on tree1
    elif score2 > tree2.r2:
        return 2  # self benefical on tree2
    else:
        return 3

def try_fusion(tree_list,finish_count):
    tree1 = tree_list[0]
    best_tree = None
    best_index = 1
    best_type = 3
    sim = 40
    set_lambda_G = 3
    model_path = r'model/update/'
    for tree_index in range(1, len(tree_list)):
        print(tree_index)
        tree2 = tree_list[tree_index]
        # if sim_compare(tree1.train_dataset, tree2.train_dataset, sim):
        #     # 相似度大于sim时返回Ture，即跳过整个不合并,剪枝，让两个数据的分布差别不大
        #     continue
        fused_train_data = torch.utils.data.ConcatDataset([tree1.train_dataset, tree2.train_dataset])
        # fused_train_dataloader = DataLoader(fused_train_data, batch_size=32, shuffle=True)
        fused_ver_data = torch.utils.data.ConcatDataset([tree1.ver_dataset, tree2.ver_dataset])
        # fused_ver_dataloader = DataLoader(fused_ver_data, batch_size=32, shuffle=True)
        model_name1 = tree1.model_name[tree1.model_name.rfind('/')+1:][:-4]
        model_name2 = tree2.model_name[tree2.model_name.rfind('/') + 1:][:-4]
        model_name = model_path+str(finish_count)+'/'+model_name1+'_f'+str(finish_count)+'_'+model_name2+'.pth'
        tree_fused = TForest_Tree(fused_train_data, fused_ver_data,model_name)
        tree_fused.train()
        # 这里求得本来是个负值 r2
        lamda_g = tree_fused.lambda_G(tree_fused.model_name)
        if -lamda_g > set_lambda_G:
            # 如果误差大于lamda_G则跳过，不合并
            continue
        score1 = tree1.predict_score(tree_fused.model_name)
        score2 = tree2.predict_score(tree_fused.model_name)
        # fusion_type 0:全合并 1,2:半合并 3:不合并
        fusion_type = check_fusion_type(score1, score2, tree1, tree2)
        print(score1, score2, fusion_type)
        if best_tree is None:
            # 一开始还没有best_tree，合并的就是best_tree
            best_tree = tree_fused
            best_type = fusion_type
        else:
            if fusion_type < best_type:
                best_tree = tree_fused
                best_index = tree_index
                best_type = fusion_type
            elif fusion_type == best_type and tree_fused.r2 > best_tree.r2:
                best_tree = tree_fused
                best_index = tree_index
    # logging.debug('Best Fusion: {} and {}'.format(str(0), str(best_index)))
    print(f'best_type:{best_type},best_index:{best_index}')
    return best_tree, best_type, best_index


def tree_update(tree_list):
    finish_count = 0
    def take_r2(elem):

        return elem.r2

    while finish_count < 5:
        print(f'finishi_count:{finish_count}')
        tree_list.sort(key = take_r2)

        # for tree_each in tree_list:
        #     print(tree_each.r2)
        #     print(tree_each.model_name)
        tree_fused, fusion_flag, best_index = try_fusion(tree_list, finish_count)

        if fusion_flag == 0:
            del tree_list[0]  # delete all two trees
            del tree_list[best_index - 1]  # the index changed when the first on deleted.
            tree_list.append(tree_fused)
            finish_count += 1
        elif fusion_flag == 1:
            del tree_list[0]  # delete the 2nd tree
            tree_list.append(tree_fused)
            finish_count += 1
        elif fusion_flag == 2:
            del tree_list[best_index]  # delete the 1st tree
            tree_list.append(tree_fused)
            finish_count += 1
        else:
            finish_count += 1
            # tree_list[0].checked = True

        # print()
        print(f'finish_count:{finish_count},结束')
        print(f'tree_list训练的r2如下')
        for tree_each in tree_list:
            print(tree_each.r2)
    return tree_list
def tforest_main():
    # logging.info("Tree Initializating")
    # tree_list = TForest_tree_initial()
    # joblib.dump(tree_list, 'initial_tree.joblib')
    tree_list = joblib.load('initial_tree.joblib')
    tree_final_list = tree_update(tree_list)
    joblib.dump(tree_final_list, 'final_tree.joblib')
    # #
    # logging.info("STP Tree Initializating")
    # stp_list = stp_tree()
    # joblib.dump(stp_list, 'stp_tree.joblib')
    # #
    # logging.info("MTP Tree Initializating")
    # mtp_list = mtp_tree()
    # joblib.dump(mtp_list, 'mtp_tree.joblib')
    # #
    # logging.info('Fcp Tree Initializating')
    # fcp_list = fcp_tree()
    # joblib.dump(fcp_list, 'fcp_tree.joblib')
    # logging.info('Fcp Tree Initializating')
    # fcp_list = fcp_tree()
    # joblib.dump(fcp_list,'fcp_tree.joblib')


def stp_tree():
    path = r'data/stp/'
    model_path = r'model/stp/'
    model_name = 'stp.pth'
    train_dataset = AOA_dataset(path + 'train/' + 'stp_train.txt')
    ver_dataset = AOA_dataset(path + 'ver/' + 'stp_ver.txt')
    tree = Tree(train_dataset, ver_dataset,model_path+model_name)
    tree.train()
    return tree

    # 输入4大场景所有的训练数据，然后训练1个tree


def mtp_tree():
    # 输入四大场景的训练数据，然后训练4个tree
    mtp_tree_list = []
    path = r'data/mtp/'
    list = ['rural', 'parkinglot', 'surface', 'highway']
    model_path = r'model/mtp/'
    for i in list:
        train_dataset = AOA_dataset(path + 'train/' + i + '_train.txt')
        ver_dataset = AOA_dataset(path + 'ver/' + i + '_ver.txt')
        tree = Tree(train_dataset, ver_dataset,model_path+i+'.pth')
        tree.train()
        mtp_tree_list.append(tree)
    return mtp_tree_list


def fcp_tree():
    fcp_tree_list = []
    path = r'data/fcp/'
    model_path = r'model/fcp/'
    for i in range(15):
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'ver/' + str(i) + '_ver.txt')
        tree = Tree(train_dataset, ver_dataset,model_path+str(i)+'.pth')
        tree.train()
        fcp_tree_list.append(tree)
    return fcp_tree_list

    # 聚完类之后重新训练几个tree


def task_selector():
    task_final = joblib.load('final_tree.joblib')
    features = []
    task_labels = []

    for index, task_each in enumerate(task_final):
        for feature_each in task_each.train_dataset.x_data:
            feature = feature_each.numpy().reshape(312, )
            features.append(feature)
            task_labels.append(index)
    # 把每个task的模型提取出来，输入数据和task label ，用模型训练
    selector = AdaBoostClassifier(random_state=36)
    selector.fit(features, task_labels)
    joblib.dump(selector, 'selector.joblib')


def task_allocated(test_dataset, selector):
    for i in test_dataset.x_data:
        feature = i.numpy().reshape(312, )
        TF_task = selector.predict([feature])[0]

        MTP_task = 1
        FCP_task = 2

        RTP_task = np.random.randint(0, 36)

    return TF_task, MTP_task, FCP_task, RTP_task

    pass

# 1
def test_main(fcp_distribution):
    # 载入4大模型和分配器
    # 改
    # tree_final = joblib.load(r'fcp.joblib')
    tree_final = joblib.load(r'/gemini/code/fcp.joblib')


    # 载入6个数据集，对应四个场景
    # 改
    # path = r'data/test/'
    path = r'/gemini/data-2/data/test/'
    test_list = ['rural-1.txt', 'rural-2.txt', 'parkinglot-1.txt', 'highway-1.txt', 'surface-1.txt', 'surface-2.txt']
    for index in range(len(test_list)):
        # 6个测试场景
        test_dataset = AOA_dataset(path + test_list[index])
        # test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=2)
        # 2
        FCP_error = []

        # 求四个task编号  TF_task,MTP_task,FCP_task,RTP_task，STP不需要求
        y_labels = test_dataset.y_data.numpy().reshape(-1, )
        for test_data in test_dataset.x_data:
            # 获取模型的序号,测试数据每行与模型对应的数据集每行做欧氏距离，求欧氏距离平均值，得到最小的欧氏距离对应的模型的序号
            # 得到模型序号从调用模型做预测，存储误差
            # 3
            FCP_dist = []
            # 4
            for fcp_index in range(len(fcp_distribution)):
                # 5
                a = fcp_distribution[fcp_index] - test_data
                # 一个训练集数据与测试集一行做差，求第二范式
                # 6
                fcp_list = torch.linalg.norm(fcp_distribution[fcp_index] - test_data, axis=1)  # [len(data),312]
                fcp_dist = torch.mean(fcp_list)
                # mtp_dist= np.mean(np.linalg.norm(mtp_feature,axis = 1))
                FCP_dist.append(fcp_dist)
            # 得到这一行数据最好的模型序号
            FCP_task = FCP_dist.index(min(FCP_dist))
            print(tree_final[FCP_task].model_name)
            model = AoACNN()
            # model_path = '/gemini/data-1/model/mtp/',不用这个了
            # 载入对应模型
            model_name = tree_final[FCP_task].model_name
            model.load_state_dict(torch.load(model_name))
            model = model.to(device)
            # 处理输入数据
            inputs = torch.unsqueeze(test_data, dim=1).to(device)
            # 预测
            with torch.no_grad():
                pre = model(inputs)
                pre = pre.item()
                FCP_error.append(pre)

        FCP_error = abs(np.array(FCP_error) - y_labels)
        print(f'{test_list[index]}的FCP_error为:{FCP_error}')
        # print(len(TF_error))
        save_path = r'/gemini/output/'
        np.savetxt(save_path + test_list[index], FCP_error, delimiter=',', fmt='%.8f')

    # return TF_error










if __name__ == '__main__':
    # 改
    # tree_list = joblib.load(r'fcp.joblib')
    tree_list = joblib.load(r'/gemini/code/fcp.joblib')
    fcp_distribution = []
    for i in range(len(tree_list)):
        loader = DataLoader(dataset=tree_list[i].train_dataset, batch_size=100000, shuffle=False)
        for data in loader:
            inputs, labels = data
            # 降维 [len(data),1,312]->[len(data),312]
            # inputs = inputs.mean(axis = 0)
            inputs = torch.squeeze(inputs)
            fcp_distribution.append(inputs)
    len(fcp_distribution)
    print(fcp_distribution[0])
    print(fcp_distribution[0].shape)
    test_main(fcp_distribution)

