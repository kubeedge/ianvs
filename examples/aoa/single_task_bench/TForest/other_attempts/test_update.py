import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# plt.style.use(['science', 'ieee'])
logging.basicConfig(level=logging.INFO)
from tree import TForest_Tree
import os
import statsmodels.api as sm
import scienceplots
import torch
def TForest_tree_initial():
    #输入每个trace数据，然后训练一个tree存到列表
    tree_list = []
    # 11,12,21,33,34,42为测试集
    for i in range(1,11):
        tree = TForest_Tree(i)
        tree.train()
        tree_list.append(tree)
    for i in range(13,21):
        tree = TForest_Tree(i)
        tree.train()
        tree_list.append(tree)
    for i in range(22,33):
        tree = TForest_Tree(i)
        tree.train()
        tree_list.append(tree)
    for i in range(35,42):
        tree = TForest_Tree(i)
        tree.train()
        tree_list.append(tree)
    return tree_list

# def try_fusion(tree_list):
#     tree1 = tree_list[0]
#     best_tree = None
#     best_index = 1
#     best_type = 3
#     for tree_index in range(1, len(tree_list)):
#         print(tree_index)
#         tree2 = tree_list[tree_index]
#         tree_fused = Tree(np.concatenate((tree1.train_features, tree2.train_features), axis=0),
#                             np.concatenate((tree1.test_features, tree2.test_features), axis=0),
#                             np.concatenate((tree1.train_labels, tree2.train_labels), axis=0),
#                             np.concatenate((tree1.test_labels, tree2.test_labels), axis=0))
#         # tree_fused = TForest_Tree(合并好的traindataset,合并好的 verificationdataset)
#         tree_fused.train()
#         score1 = tree1.predict_score(tree_fused.model)
#         score2 = tree2.predict_score(tree_fused.model)
#         score3 = tree_fused.lamda()
#         fusion_type = check_fusion_type(score1, score2,score3, tree1, tree2)
#         print(score1, score2,score3, fusion_type)
#         if best_tree is None:
#             best_tree = tree_fused
#             best_type = fusion_type
#         else:
#             if fusion_type < best_type:
#                 best_tree = tree_fused
#                 best_index = tree_index
#                 best_type = fusion_type
#             elif fusion_type == best_type and tree_fused.r2 > best_tree.r2:
#                 best_tree = tree_fused
#                 best_index = tree_index
#     logging.debug('Best Fusion: {} and {}'.format(str(0), str(best_index)))
#
#     return best_tree, best_type, best_index


# def tree_update(tree_list):
#     finish_count = 0
#
#     def take_r2(elem):
#         if elem.checked is True:
#             return 1
#         else:
#             return elem.r2
#
#     while finish_count < 20:
#
#         tree_list.sort(key=take_r2)
#         logging.info("Before Updating!")
#         for tree_each in tree_list:
#             print(tree_each.r2)
#         tree_fused, fusion_flag, best_index = try_fusion(tree_list)
#         if fusion_flag == 0:
#             del tree_list[0]  # delete all two trees
#             del tree_list[best_index - 1]  # the index changed when the first on deleted.
#             tree_list.append(tree_fused)
#             finish_count = 0
#         elif fusion_flag == 1:
#             del tree_list[0]  # delete the 2nd tree
#             tree_list.append(tree_fused)
#             finish_count = 0
#         elif fusion_flag == 2:
#             del tree_list[best_index]  # delete the 1st tree
#             tree_list.append(tree_fused)
#             finish_count = 0
#         else:
#             finish_count += 1
#             tree_list[0].checked = True
#
#         logging.info("Tree Updating, Fusion Flag: " + str(fusion_flag))
#         for tree_each in tree_list:
#             print(tree_each.r2)
#     return tree_list


def tforest_main():
    logging.info("Tree Initializating")
    tree_list = TForest_tree_initial()
    joblib.dump(tree_list, 'initial_tree.joblib')

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
    set_lambda_G = 1
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
        lamda_g = tree_fused.lambda_G()
        # if lambda_G < lamda_g:
        #     # 如果误差大于lamda_G则跳过，不合并
        #     continue
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
        for tree_each in tree_list:
            print(tree_each.r2)
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
        for tree_each in tree_list:
            print(tree_each.r2)
    return tree_list

if __name__ == '__main__':
    # tforest_main()
    tree_list = joblib.load('test_init_tree.joblib')
    # print(len(tree_list))
    # tree_list = tree_update(tree_list)


    tree_final_list = tree_update(tree_list)


    print(1)

