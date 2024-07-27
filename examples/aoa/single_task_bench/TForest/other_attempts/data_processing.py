# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
def read_files():
    """该函数用于读取对应文件夹下各txt文件的名字"""
    path = r'data/processed/'
    files = os.listdir(path)


    file_names = []
    for file in files:
        if file.split('.')[-1] == 'txt':  # 如果不是txt文件就跳过
            file_names.append(file)
    return path, file_names
def mixed_file(path,file_list,t_v):
    """该函数用于合并刚才读取的各文件
    输入：文件路径，read_files()返回的文件名
    输出：一个合并后的文件"""
    content = ''
    # for file_name in files:
    for i in file_list:
        with open(path+ str(i)+'_'+t_v+'.txt', 'r', encoding='utf-8') as file:
            content = content + file.read()
            file.close()

    with open(r'data/general_data/' + 'gen_'+t_v+'.txt', 'a', encoding='utf-8') as file:
        file.write(content)
        content = ''
        file.close()
# 读取文件，随机 9：1 分数据，数据保存到两个文件里
def tforest_data(ord_num):
    path = r'data/processed/'

    data = np.loadtxt(path+str(ord_num)+'.txt', delimiter=',')
    features = data[:, 10:-1]
    labels = data[:, 5]
    train_features, ver_features, train_labels, ver_labels = train_test_split(features, labels,train_size=0.9,shuffle=True,random_state=42)
    # train_features和train_labels合并一个test,再保存到train中
    train_data = np.hstack((train_features,train_labels.reshape(-1,1)))
    np.savetxt('data/train/'+str(ord_num)+'_train.txt',train_data , delimiter=',', fmt='%.6f')
    # ver_features和ver
    ver_data = np.hstack((ver_features,ver_labels.reshape(-1,1)))
    np.savetxt('data/verification/' + str(ord_num) + '_ver.txt', ver_data, delimiter=',', fmt='%.6f')


def data_process(ord_num,path,situ):
    data = np.loadtxt(r'data/processed/' + str(ord_num) + '.txt', delimiter=',')
    features = data[:, 10:-1]
    labels = data[:, 5]
    proce_data = np.hstack((features, labels.reshape(-1, 1)))
    np.savetxt(path + str(ord_num) + '_'+situ+'.txt', proce_data, delimiter=',', fmt='%.6f')



def mixed_file_2(path,file_list,str_list):
    """该函数用于合并刚才读取的各文件
    输入：文件路径，read_files()返回的文件名
    输出：一个合并后的文件"""
    file_path = r'data/processed_data/'
    content = ''
    # for file_name in files:
    for i in file_list:
        with open(file_path+ str(i)+'_'+'.txt', 'r', encoding='utf-8') as file:
            content = content + file.read()
            file.close()

    with open(path + str_list+'.txt', 'a', encoding='utf-8') as file:
        file.write(content)
        content = ''
        file.close()
def stp_data():
    path = r'data/stp/stp_train.txt'

    data = np.loadtxt(path, delimiter=',')
    features = data[:, :-1]
    labels = data[:, -1]
    train_features, ver_features, train_labels, ver_labels = train_test_split(features, labels,train_size=0.9,shuffle=True,random_state=42)
    # train_features和train_labels合并一个test,再保存到train中
    train_data = np.hstack((train_features,train_labels.reshape(-1,1)))
    np.savetxt('data/stp/train/stp_train.txt',train_data , delimiter=',', fmt='%.6f')
    # ver_features和ver
    ver_data = np.hstack((ver_features,ver_labels.reshape(-1,1)))
    np.savetxt('data/stp/ver/stp_ver.txt', ver_data, delimiter=',', fmt='%.6f')



def mtp_data(situ):
    path = r'data/mtp/'+situ+'_train.txt'
    data = np.loadtxt(path, delimiter=',')
    features = data[:, :-1]
    labels = data[:, -1]
    train_features, ver_features, train_labels, ver_labels = train_test_split(features, labels,train_size=0.9,shuffle=True,random_state=42)
    # train_features和train_labels合并一个test,再保存到train中
    train_data = np.hstack((train_features,train_labels.reshape(-1,1)))
    np.savetxt('data/mtp/train/'+situ+'_train.txt',train_data , delimiter=',', fmt='%.6f')
    # ver_features和ver
    ver_data = np.hstack((ver_features,ver_labels.reshape(-1,1)))
    np.savetxt('data/mtp/ver/'+situ+'_ver.txt', ver_data, delimiter=',', fmt='%.6f')


def fcp_data():
    for i in range(15):
        filename = r'data/fcp/'+str(i)+'.txt'
        data = np.loadtxt(filename,delimiter=',')
        features = data[:,:-1]
        labels = data[:,-1]
        train_features, ver_features, train_labels, ver_labels = train_test_split(features, labels, train_size=0.9,
                                                                                  shuffle=True, random_state=42)
        # train_features和train_labels合并一个test,再保存到train中
        train_data = np.hstack((train_features, train_labels.reshape(-1, 1)))
        np.savetxt('data/fcp/train/' + str(i)+ '_train.txt', train_data, delimiter=',', fmt='%.6f')
        # ver_features和ver
        ver_data = np.hstack((ver_features, ver_labels.reshape(-1, 1)))
        np.savetxt('data/fcp/ver/' +  str(i) + '_ver.txt', ver_data, delimiter=',', fmt='%.6f')

def test_data():
    # testdata也是处理过的
    path = r'data/test/'

    test_list = ['rural-1.txt', 'rural-2.txt', 'parkinglot-1.txt', 'highway-1.txt', 'surface-1.txt','surface-2.txt']
    for i in test_list:

        data = np.loadtxt(path + i, delimiter=',')
        features = data[:, 10:-1]
        labels = data[:, 5]
    # train_features, ver_features, train_labels, ver_labels = train_test_split(features, labels, train_size=0.9,
    #                                                                           shuffle=True, random_state=42)
    # train_features和train_labels合并一个test,再保存到train中
        test_data = np.hstack((features, labels.reshape(-1, 1)))
        np.savetxt(path+ 'test_data/'+ i , test_data, delimiter=',', fmt='%.6f')



if __name__ == '__main__':

    # file_list = [1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,35,36,37,38,39,40,41,42]
    # path = r'data/verification/'
    # mixed_file(path,file_list,'ver')
    # stp_train_list =[1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,35,36,37,38,39,40,41,42]
    # stp_path = r'data/stp/train'

    # stp_test_list=[]
    # mtp_train_list = []
    #stp数据，所有数据合并成一个
    #fcp数据，聚类成几个
    #mtp数据，4个

    # for i in range(1,43):
    #     data_process(i,r'data/processed_data/','')
    # stp
    # 1-10  10
    # 13-20 8
    # 22-32 11
    # 35-41 7
    # stp_file_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #                     13, 14, 15, 16, 17, 18, 19, 20,
    #                      22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32,
    #                      35, 36, 37, 38, 39, 40, 41]
    # stp_path = r'data/stp/'
    # mixed_file_2(stp_path,stp_file_list,'stp_train')
    # stp_data()

    # mtp

    # rural_file_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # mtp_path =r'data/mtp/'
    # mixed_file_2( mtp_path, rural_file_list, 'rural_train')
    #
    # parkinglot_file_list = [13, 14, 15, 16, 17, 18, 19, 20]
    # mtp_path = r'data/mtp/'
    # mixed_file_2(mtp_path, parkinglot_file_list, 'parkinglot_train')
    #
    # surface_file_list = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32]
    # mtp_path = r'data/mtp/'
    # mixed_file_2(mtp_path, surface_file_list, 'surface_train')
    #
    # highway_file_list = [35, 36, 37, 38, 39, 40, 41]
    # mtp_path = r'data/mtp/'
    # mixed_file_2(mtp_path,highway_file_list, 'highway_train')
    #
    # list = ['rural', 'parkinglot','surface','highway']
    # for i in list:
    #     mtp_data(i)

    test_data()
    #fcp
    # path = r'data/fcp/'
    # data = np.loadtxt(path+'fcp.txt', delimiter=',')
    # all_features = data[:, :-1]
    # all_labels = data[:, -1]
    #
    # all_features = pd.DataFrame(all_features)
    # # 数据标准化
    # z_scaler = preprocessing.StandardScaler()
    # data_fcp1 = z_scaler.fit_transform(all_features)
    # # # 要不要转成dataframe另说
    # # data_fcp1 = pd.DataFrame(data_fcp1)
    # # 降维
    # pca = PCA(n_components=6)
    # data_fcp1 = pca.fit_transform(data_fcp1)
    # # 数据归一化
    # minmax_scale = preprocessing.MinMaxScaler().fit(data_fcp1)
    # data_fcp2 = minmax_scale.transform(data_fcp1)
    #
    # k_means = KMeans(init='k-means++', n_init="auto", n_clusters=15,
    #                  random_state=10)  # init='k-means++'表示用kmeans++的方法来选择初始质数 n_clusters=8表示数据聚为8类 max_iter=500表示最大的迭代次数是500(缺省300)
    # k_means.fit(data_fcp2)
    # labels = k_means.labels_
    # y_pred = k_means.fit_predict(data_fcp2)
    # print(y_pred)
    #
    # all_features['labels'] = labels
    # all_features['label'] = all_labels
    # path = "data/fcp/"
    # for i in range(15):
    #     m = all_features.loc[all_features['labels'] == i]
    #     filepath = path + str(i) + ".txt"
    #     print(filepath)
    #     m = m.drop(['labels'], axis=1)
    #
    #     m.to_csv(path + str(i) + ".txt", index=False, header=False)
    # data = np.loadtxt(r'data/fcp/1.txt',delimiter=',')
    # # fcp_data = np.loadtxt(r'data/fcp/fcp.txt',delimiter=',')
    # fcp_data()
    # print(1)
    #





"""
    # 绘制图像判断折点
    K = range(1, 37)

    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_fcp2)
        # 计算各个点分别到k个质心的距离,取最小值作为其到所属质心的距离,并计算这些点到各自所属质心距离的平均距离
        meandistortions.append(
            sum(
                np.min(cdist(data_fcp2, kmeans.cluster_centers_, 'euclidean'), axis=1)
            ) / data_fcp2.shape[0]
        )
    plt.plot(K, meandistortions, 'bx--')
    plt.xlabel('k')
    plt.savefig("Gravel figure_pca_2.png")
    plt.show()

"""













# 读取文件，随机 9：1 分数据，数据保存到两个文件里