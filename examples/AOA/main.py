from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
import torch
from torch.utils.data import DataLoader
import scienceplots
import statsmodels.api as sm
import os
from tree import TForest_Tree, AOA_dataset, Tree
import logging
import numpy as np
import joblib
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee'])
logging.basicConfig(level=logging.INFO)
# from tqdm import tqdm
# plt.style.use(['science', 'ieee'])


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
        tree = TForest_Tree(train_dataset, ver_dataset, model_path+model_name)
        tree.train()
        tree_list.append(tree)
    for i in range(13, 21):
        model_path = r'model/tforest/'
        model_name = str(i) + '.pth'
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'verification/' + str(i) + '_ver.txt')
        tree = TForest_Tree(train_dataset, ver_dataset, model_path+model_name)
        tree.train()
        tree_list.append(tree)
    for i in range(22, 33):
        model_path = r'model/tforest/'
        model_name = str(i) + '.pth'
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'verification/' + str(i) + '_ver.txt')
        tree = TForest_Tree(train_dataset, ver_dataset, model_path+model_name)
        tree.train()
        tree_list.append(tree)
    for i in range(35, 42):
        model_path = r'model/tforest/'
        model_name = str(i) + '.pth'
        train_dataset = AOA_dataset(path + 'train/' + str(i) + '_train.txt')
        ver_dataset = AOA_dataset(path + 'verification/' + str(i) + '_ver.txt')
        tree = TForest_Tree(train_dataset, ver_dataset, model_path+model_name)
        tree.train()
        tree_list.append(tree)

    return tree_list


# def tree_update(tree_list):

def sim_compare(dataset1, dataset2, sim=30):
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


def try_fusion(tree_list, finish_count):
    tree1 = tree_list[0]
    best_tree = None
    best_index = 1
    best_type = 3
    sim = 40
    set_lambda_G = 3
    model_path = r'/gemini/output/'
    for tree_index in range(1, len(tree_list)):
        print(tree_index)
        tree2 = tree_list[tree_index]
        # if sim_compare(tree1.train_dataset, tree2.train_dataset, sim):
        #     # 相似度大于sim时返回Ture，即跳过整个不合并,剪枝，让两个数据的分布差别不大
        #     continue
        fused_train_data = torch.utils.data.ConcatDataset(
            [tree1.train_dataset, tree2.train_dataset])
        # fused_train_dataloader = DataLoader(fused_train_data, batch_size=32, shuffle=True)
        fused_ver_data = torch.utils.data.ConcatDataset(
            [tree1.ver_dataset, tree2.ver_dataset])
        # fused_ver_dataloader = DataLoader(fused_ver_data, batch_size=32, shuffle=True)
        model_name1 = tree1.model_name[tree1.model_name.rfind('/')+1:][:-4]
        model_name2 = tree2.model_name[tree2.model_name.rfind('/') + 1:][:-4]
        model_name = model_path + \
            str(finish_count)+'/'+model_name1+'_f' + \
            str(finish_count)+'_'+model_name2+'.pth'
        tree_fused = TForest_Tree(fused_train_data, fused_ver_data, model_name)
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
        tree_list.sort(key=take_r2)

        # for tree_each in tree_list:
        #     print(tree_each.r2)
        #     print(tree_each.model_name)
        tree_fused, fusion_flag, best_index = try_fusion(
            tree_list, finish_count)

        if fusion_flag == 0:
            del tree_list[0]  # delete all two trees
            # the index changed when the first on deleted.
            del tree_list[best_index - 1]
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

    logging.info("Tree Initializating")
    tree_list = TForest_tree_initial()
    joblib.dump(tree_list, 'initial_tree.joblib')

    path = 'output/'
    print(os.path.exists(path))
    for i in range(5):
        os.mkdir(path + str(i))
    print('已生成文件夹')
    joblib_path = r'initial.joblib'
    tree_list = joblib.load(joblib_path)
    tree_final_list = tree_update(tree_list)
    joblib.dump(tree_final_list, 'output/final_tree.joblib')

    # tree_list = joblib.load('initial_tree.joblib')
    # tree_final_list = tree_update(tree_list)
    # joblib.dump(tree_final_list, 'final_tree.joblib')

    # stp
    logging.info("STP Tree Initializating")
    stp_list = stp_tree()
    joblib.dump(stp_list, 'stp_tree.joblib')
    # mtp
    logging.info("MTP Tree Initializating")
    mtp_list = mtp_tree()
    joblib.dump(mtp_list, 'mtp_tree.joblib')
    # fcp
    logging.info('Fcp Tree Initializating')
    fcp_list = fcp_tree()
    joblib.dump(fcp_list, 'fcp_tree.joblib')


def stp_tree():
    path = r'data/stp/'
    model_path = r'model/stp/'
    model_name = 'stp.pth'
    train_dataset = AOA_dataset(path + 'train/' + 'stp_train.txt')
    ver_dataset = AOA_dataset(path + 'ver/' + 'stp_ver.txt')
    tree = Tree(train_dataset, ver_dataset, model_path+model_name)
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
        tree = Tree(train_dataset, ver_dataset, model_path+i+'.pth')
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
        tree = Tree(train_dataset, ver_dataset, model_path+str(i)+'.pth')
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


def test_main():
    # 载入4大模型和分配器
    tree_final = joblib.load('initial_tree.joblib')
    fcp_tree = joblib.load('fcp_tree.joblib')
    mtp_tree = joblib.load('mtp_tree.joblib')
    stp_tree = joblib.load('stp_tree.joblib')
    


def cre_error_data():
    # tforest
    tree_list = joblib.load(r'joblib/tf_final.joblib')
    tf_distribution = []
    for i in range(len(tree_list)):
        loader = DataLoader(
            dataset=tree_list[i].train_dataset, batch_size=100000, shuffle=False)
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

    # stp
    tree_list = joblib.load(r'joblib/stp.joblib')

    stp_distribution = []

    loader = DataLoader(dataset=tree_list.train_dataset,
                        batch_size=1000000, shuffle=False)
    for data in loader:
        inputs, labels = data
        # 降维 [len(data),1,312]->[len(data),312]
        # inputs = inputs.mean(axis = 0)
        inputs = torch.squeeze(inputs)
        stp_distribution.append(inputs)
    len(stp_distribution)
    print(stp_distribution[0])
    print(stp_distribution[0].shape)
    test_main(stp_distribution)

    # mtp
    tree_list = joblib.load(r'joblib/mtp.joblib')
    mtp_distribution = []
    for i in range(len(tree_list)):
        loader = DataLoader(
            dataset=tree_list[i].train_dataset, batch_size=100000, shuffle=False)
        for data in loader:
            inputs, labels = data
            # 降维 [len(data),1,312]->[len(data),312]
            # inputs = inputs.mean(axis = 0)
            inputs = torch.squeeze(inputs)
            mtp_distribution.append(inputs)
    len(mtp_distribution)
    print(mtp_distribution[0])
    print(mtp_distribution[0].shape)
    test_main(mtp_distribution)
    # fcp
    tree_list = joblib.load(r'joblib/fcp.joblib')
    fcp_distribution = []
    for i in range(len(tree_list)):
        loader = DataLoader(
            dataset=tree_list[i].train_dataset, batch_size=100000, shuffle=False)
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


def cdf_draw_each(data, label, color='#8ea3c2', marker='*', linestyle='-', markersize=3):
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(min(data), max(data))
    y = ecdf(x)
    # plt.step(x, y, label=label, color=color, markersize=markersize, marker=marker, linestyle=linestyle)
    plt.step(x, y, label=label)


def draw_pic():
    test_list = ['rural-1.txt', 'rural-2.txt', 'parkinglot-1.txt',
                 'highway-1.txt', 'surface-1.txt', 'surface-2.txt']
    error_data_path = r'error_data/'
    for i in test_list:
        # 获取某测试数据的误差
        TF_error = np.loadtxt(error_data_path + 'tf/' + i, delimiter=',')
        STP_error = np.loadtxt(error_data_path + 'stp/' + i, delimiter=',')
        FCP_error = np.loadtxt(error_data_path + 'fcp/' + i, delimiter=',')
        MTP_error = np.loadtxt(error_data_path + 'mtp/' + i, delimiter=',')
        RTP_error = np.loadtxt(error_data_path + 'rtp/' + i, delimiter=',')

        # 绘图
        cdf_draw_each(TF_error, 'TForest')
        cdf_draw_each(STP_error, 'STP')
        cdf_draw_each(FCP_error, 'FCP')
        cdf_draw_each(MTP_error, 'MTP')
        cdf_draw_each(RTP_error, 'RTP')

        plt.title('Mean error of Aoa on ' + i)
        plt.legend()
        file_str = 'Mean error of Aoa on ' + i+'.png'
        plt.savefig('pic/'+file_str)
        plt.show()


if __name__ == '__main__':

    # 首先训练模型
    tforest_main()

    # 生成误差数据
    cre_error_data()
    # 绘图
    draw_pic()
