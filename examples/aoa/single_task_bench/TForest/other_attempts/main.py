import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt

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
def TForest_tree_initial():
    #输入每个trace数据，然后训练一个tree存到列表
    tree_list = []
    # 11,12,21,33,34,42为测试集
    path = r'data/'
    # 后面再改成11啊
    for i in range(1,11):
        train_dataset = AOA_dataset(path+'train/'+str(i)+'_train.txt')
        ver_dataset = AOA_dataset(path+'verification/'+str(i)+'_ver.txt')
        tree = TForest_Tree(train_dataset,ver_dataset)
        tree.train()
        tree_list.append(tree)
    for i in range(13,21):
        train_dataset = AOA_dataset(path+'train/'+str(i)+'_train.txt')
        ver_dataset = AOA_dataset(path+'verification/'+str(i)+'_ver.txt')
        tree = TForest_Tree(train_dataset,ver_dataset)
        tree.train()
        tree_list.append(tree)
    for i in range(22,33):
        train_dataset = AOA_dataset(path+'train/'+str(i)+'_train.txt')
        ver_dataset = AOA_dataset(path+'verification/'+str(i)+'_ver.txt')
        tree = TForest_Tree(train_dataset,ver_dataset)
        tree.train()
        tree_list.append(tree)
    for i in range(35,42):
        train_dataset = AOA_dataset(path+'train/'+str(i)+'_train.txt')
        ver_dataset = AOA_dataset(path+'verification/'+str(i)+'_ver.txt')
        tree = TForest_Tree(train_dataset,ver_dataset)
        tree.train()
        tree_list.append(tree)

    return tree_list

# def tree_update(tree_list):

def sim_compare(dataset1,dataset2,sim=30):


    mean1 = torch.mean(dataset1.x_data, dim=0)
    mean2 = torch.mean(dataset2.x_data, dim=0)
    dist = torch.linalg.norm(mean1 - mean2).item()
    if dist > sim:
        return True
    else:
        return False

# def general_condition():
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

def try_fusion(tree_list):
    tree1 = tree_list[0]
    best_tree = None
    best_index = 1
    best_type = 3
    sim = 40
    set_lambda_G = 3
    for tree_index in range(1, len(tree_list)):
        print(tree_index)
        tree2 = tree_list[tree_index]
        if sim_compare(tree1.train_dataset,tree2.train_dataset,sim):
            # 相似度大于sim时返回Ture，即跳过整个不合并
            continue
        fused_train_data = torch.utils.data.ConcatDataset([tree1.train_dataset, tree2.train_dataset])
        # fused_train_dataloader = DataLoader(fused_train_data, batch_size=32, shuffle=True)
        fused_ver_data = torch.utils.data.ConcatDataset([tree1.ver_dataset, tree2.ver_dataset])
        # fused_ver_dataloader = DataLoader(fused_ver_data, batch_size=32, shuffle=True)
        tree_fused = TForest_Tree(fused_train_data,fused_ver_data)
        tree_fused.train()
        # 这里求得本来是个负值 r2
        lamda_g = tree_fused.lambda_G(tree_fused.model)
        if  -lamda_g>set_lambda_G:
            # 如果误差大于lamda_G则跳过，不合并
            continue
        score1 = tree1.predict_score(tree_fused.model)
        score2 = tree2.predict_score(tree_fused.model)
        fusion_type = check_fusion_type(score1, score2, tree1, tree2)
        print(score1, score2, fusion_type)
        if best_tree is None:
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
    logging.debug('Best Fusion: {} and {}'.format(str(0), str(best_index)))

    return best_tree, best_type, best_index

def tree_update(tree_list):
    finish_count = 0

    def take_r2(elem):
        if elem.checked is True:
            return 1
        else:
            return elem.r2

    while finish_count < 20:



        tree_list.sort(key=take_r2)
        logging.info("Before Updating!")
        # for tree_each in tree_list:
        #     print(tree_each.r2)
        tree_fused, fusion_flag, best_index = try_fusion(tree_list)
        if fusion_flag == 0:
            del tree_list[0]  # delete all two trees
            del tree_list[best_index - 1]  # the index changed when the first on deleted.
            tree_list.append(tree_fused)
            finish_count = 0
        elif fusion_flag == 1:
            del tree_list[0]  # delete the 2nd tree
            tree_list.append(tree_fused)
            finish_count = 0
        elif fusion_flag == 2:
            del tree_list[best_index]  # delete the 1st tree
            tree_list.append(tree_fused)
            finish_count = 0
        else:
            finish_count += 1
            tree_list[0].checked = True

        logging.info("Tree Updating, Fusion Flag: " + str(fusion_flag))
        print(f'finish_count:{finish_count}')
        for tree_each in tree_list:
            print(tree_each.r2,end='   ')
    return tree_list


def tforest_main():

    logging.info("Tree Initializating")
    tree_list = TForest_tree_initial()
    joblib.dump(tree_list, 'initial_tree.joblib')
    tree_list = joblib.load('initial_tree.joblib')
    tree_final_list = tree_update(tree_list)
    joblib.dump(tree_final_list, 'final_tree.joblib')




    logging.info("STP Tree Initializating")
    stp_list = stp_tree()
    joblib.dump(stp_list, 'stp_tree.joblib')

    logging.info("MTP Tree Initializating")
    mtp_list = mtp_tree()
    joblib.dump(mtp_list,'mtp_tree.joblib')

    logging.info('Fcp Tree Initializating')
    fcp_list = fcp_tree()
    joblib.dump(fcp_list,'fcp_tree.joblib')
    # logging.info('Fcp Tree Initializating')
    # fcp_list = fcp_tree()
    # joblib.dump(fcp_list,'fcp_tree.joblib')


def stp_tree():
    path = r'data/stp/'
    train_dataset = AOA_dataset(path + 'train/' + 'stp_train.txt')
    ver_dataset = AOA_dataset(path + 'ver/'+'stp_ver.txt')
    tree = Tree(train_dataset, ver_dataset)
    tree.train()
    return tree

    #输入4大场景所有的训练数据，然后训练1个tree


def mtp_tree():
    # 输入四大场景的训练数据，然后训练4个tree
    mtp_tree_list = []
    path = r'data/mtp/'
    list = ['rural', 'parkinglot','surface','highway']
    for i in list:

        train_dataset = AOA_dataset(path + 'train/' + i+'_train.txt')
        ver_dataset = AOA_dataset(path + 'ver/' + i+'_ver.txt')
        tree = Tree(train_dataset, ver_dataset)
        tree.train()
        mtp_tree_list.append(tree)
    return mtp_tree_list



def fcp_tree():
    fcp_tree_list = []
    path = r'data/fcp/'
    for i in range(15):
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'ver/' + str(i) + '_ver.txt')
        tree = Tree(train_dataset, ver_dataset)
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
            feature = feature_each.numpy().reshape(312,)
            features.append(feature)
            task_labels.append(index)
    #把每个task的模型提取出来，输入数据和task label ，用模型训练
    selector=AdaBoostClassifier(random_state=36)
    selector.fit(features, task_labels)
    joblib.dump(selector, 'selector.joblib')

def task_allocated(test_dataset,selector):
    for i in test_dataset.x_data:
        feature = i.numpy().reshape(312,)
        TF_task = selector.predict([feature])[0]

        MTP_task = 1
        FCP_task = 2

        RTP_task = np.random.randint(0,36)

    return TF_task,MTP_task,FCP_task,RTP_task

    pass


def test_main():
    # 载入4大模型和分配器
    tree_final = joblib.load('final_tree.joblib')
    fcp_tree = joblib.load('fcp_tree.joblib')
    mtp_tree = joblib.load('mtp_tree.joblib')
    stp_tree = joblib.load('stp_tree.joblib')
    selector = joblib.load('selector.joblib')

    # 载入6个数据集，对应四个场景
    path = r'data/test/test_data/'
    test_list = ['rural-1.txt','rural-2.txt','parkinglot-1.txt','highway-1.txt','surface-1.txt','surface-2.txt']
    for index in tqdm(range(len(test_list))):
        # 6个测试场景
        test_dataset = AOA_dataset(path + test_list[index])
        # test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=2)



        TF_error = []
        STP_error = []
        MTP_error = []
        FCP_error = []
        RTP_error = []
        # 求四个task编号  TF_task,MTP_task,FCP_task,RTP_task，STP不需要求
        y_labels = test_dataset.y_data.numpy().reshape(1,-1)
        for i in test_dataset.x_data:
            feature = i.numpy().reshape(312, )
            # TF_task
            TF_task = selector.predict([feature])[0]
            """
            # MTP_task
            Mtp_dist = []
            for mtp_index in mtp_tree:
                mtp_feature = mtp_index.train_dataset.x_data.numpy().reshape(-1,312)
                # mtp_feature2 = mtp_feature.reshape(12363,312)
                #feature与每行求欧氏距离，再求绝对值
                mtp_list = np.linalg.norm(mtp_feature-feature,axis = 1)
                mtp_dist = np.mean(mtp_list)
                # mtp_dist= np.mean(np.linalg.norm(mtp_feature,axis = 1))
                Mtp_dist.append(mtp_dist)
                # 是其中最小值的位置
            MTP_task = Mtp_dist.index(min(Mtp_dist))
            # FCP_task
            FCP_dist = []
            for fcp_index in fcp_tree:
                fcp_feature = fcp_index.train_dataset.x_data.numpy().reshape(-1,312)
                # mtp_feature2 = mtp_feature.reshape(12363,312)
                #feature与每行求欧氏距离，再求绝对值
                fcp_list = np.linalg.norm(fcp_feature-feature,axis = 1)
                fcp_dist = np.mean(fcp_list)
                # mtp_dist= np.mean(np.linalg.norm(mtp_feature,axis = 1))
                FCP_dist.append(fcp_dist)
                # 是其中最小值的位置
            FCP_task = FCP_dist.index(min(FCP_dist))
            # RTP_task
            RTP_task = np.random.randint(0, 36)
            # TF_task,MTP_task,FCP_task,RTP_task
            """


            # 预测值
            # TForest
            inputs = torch.unsqueeze(i, dim=1).to(device)
            with torch.no_grad():
                model = tree_final[TF_task].model.to(device)
                pre = model(inputs)
                pre = pre.item()
                TF_error.append(pre)
            # TF_error = np.array(TF_error)
            """
             # stp预测
            with torch.no_grad():
                model = stp_tree.model.to(device)
                pre = model(inputs).item()
                STP_error.append(pre)
            # mtp预测
            with torch.no_grad():
                model = mtp_tree[MTP_task].model.to(device)
                pre = model(inputs).item()
                MTP_error.append(pre)
            # fcp预测
            with torch.no_grad():
                model = fcp_tree[FCP_task].model.to(device)
                pre = model(inputs).item()
                FCP_error.append(pre)
            # rtp预测
            with torch.no_grad():
                model = tree_final[RTP_task].model.to(device)
                pre = model(inputs).item()
                RTP_error.append(pre)
            
            """


        TF_error = np.array(TF_error)
        # STP_error = np.array(STP_error)
        # MTP_error = np.array(MTP_error)
        # FCP_error = np.array(FCP_error)
        # RTP_error = np.array(RTP_error)

        TF_error = np.array(TF_error)-y_labels
        # STP_error = np.array(STP_error)-y_labels
        # MTP_error = np.array(MTP_error)-y_labels
        # FCP_error = np.array(FCP_error)-y_labels
        # RTP_error = np.array(RTP_error)-y_labels




            # pass
        # tf预测
        # stp预测
        # mtp预测
        # fcp预测

    pass

if __name__ == '__main__':
    # tforest_main()
    # test里面开始，构建test的数据集
    # task_selector()
    test_main()
    print(1)

