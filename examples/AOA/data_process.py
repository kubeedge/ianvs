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
"""
I：
O：
读取数据文件，获取features = data[:, 10:-1],labels = data[:, 5],横向合并,保存到data/processed_data/
例如1.txt,2.txt
"""
def data_process(ord_num,path,situ):
    data = np.loadtxt(r'data/processed/' + str(ord_num) + '.txt', delimiter=',')
    features = data[:, 10:-1]
    labels = data[:, 5]
    proce_data = np.hstack((features, labels.reshape(-1, 1)))
    np.savetxt(path + str(ord_num) +situ+'.txt', proce_data, delimiter=',', fmt='%.6f')

def data_preprocessing():
    for i in range(1, 43):
        data_process(i, r'data/processed_data/', '')

"""
mixed_file_2(path,file_list,str_list):

I:path,file_list,str_list
保存文件位置，需要合并文件的序号，保存的文件名
O：合并后文件
"""

def mixed_file_2(path,file_list,str_list):
    """该函数用于合并刚才读取的各文件
    输入：文件路径，read_files()返回的文件名
    输出：一个合并后的文件"""
    file_path = r'data/processed_data/'
    content = ''
    # for file_name in files:
    for i in file_list:
        with open(file_path+ str(i)+'.txt', 'r', encoding='utf-8') as file:
            content = content + file.read()
            file.close()

    with open(path + str_list+'.txt', 'a', encoding='utf-8') as file:
        file.write(content)
        content = ''
        file.close()

"""
读取stp的训练文件，随机9:1划分为train和ver，保存到data/stp/train/和data/stp/ver
"""
def stp_data():
    path = r'data/stp/stp.txt'

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



def stp_preprocessing():
    # stp
    # 1-10  10
    # 13-20 8
    # 22-32 11
    # 35-41 7
    stp_file_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    13, 14, 15, 16, 17, 18, 19, 20,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32,
                    35, 36, 37, 38, 39, 40, 41]
    stp_path = r'data/stp/'
    mixed_file_2(stp_path,stp_file_list,'stp')
    stp_data()
"""
合并mtp四种场景下的数据文件，然后9:1划分为训练集和测试集
"""

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



def mtp_preprocessing():
    # mtp
    rural_file_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mtp_path =r'data/mtp/'
    mixed_file_2( mtp_path, rural_file_list, 'rural_train')

    parkinglot_file_list = [13, 14, 15, 16, 17, 18, 19, 20]
    mtp_path = r'data/mtp/'
    mixed_file_2(mtp_path, parkinglot_file_list, 'parkinglot_train')

    surface_file_list = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32]
    mtp_path = r'data/mtp/'
    mixed_file_2(mtp_path, surface_file_list, 'surface_train')

    highway_file_list = [35, 36, 37, 38, 39, 40, 41]
    mtp_path = r'data/mtp/'
    mixed_file_2(mtp_path,highway_file_list, 'highway_train')

    list = ['rural', 'parkinglot','surface','highway']
    for i in list:
        mtp_data(i)


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



def fcp_preprocessing():


    # fcp
    path = r'data/fcp/'
    data = np.loadtxt(path+'fcp.txt', delimiter=',')
    all_features = data[:, :-1]
    all_labels = data[:, -1]

    all_features = pd.DataFrame(all_features)
    # 数据标准化
    z_scaler = preprocessing.StandardScaler()
    data_fcp1 = z_scaler.fit_transform(all_features)
    # # 要不要转成dataframe另说
    # data_fcp1 = pd.DataFrame(data_fcp1)
    # 降维
    pca = PCA(n_components=6)
    data_fcp1 = pca.fit_transform(data_fcp1)
    # 数据归一化
    minmax_scale = preprocessing.MinMaxScaler().fit(data_fcp1)
    data_fcp2 = minmax_scale.transform(data_fcp1)

    k_means = KMeans(init='k-means++', n_init="auto", n_clusters=15,
                     random_state=10)  # init='k-means++'表示用kmeans++的方法来选择初始质数 n_clusters=8表示数据聚为8类 max_iter=500表示最大的迭代次数是500(缺省300)
    k_means.fit(data_fcp2)
    labels = k_means.labels_
    y_pred = k_means.fit_predict(data_fcp2)
    print(y_pred)

    all_features['labels'] = labels
    all_features['label'] = all_labels
    path = "data/fcp/"
    for i in range(15):
        m = all_features.loc[all_features['labels'] == i]
        filepath = path + str(i) + ".txt"
        print(filepath)
        m = m.drop(['labels'], axis=1)

        m.to_csv(path + str(i) + ".txt", index=False, header=False)
    data = np.loadtxt(r'data/fcp/1.txt',delimiter=',')
    # fcp_data = np.loadtxt(r'data/fcp/fcp.txt',delimiter=',')
    fcp_data()


"""
tforest的数据

"""
def tforest_data(ord_num):
    path = r'data/processed_data/'

    data = np.loadtxt(path+str(ord_num)+'.txt', delimiter=',')
    features = data[:, :-1]
    labels = data[:, -1]
    train_features, ver_features, train_labels, ver_labels = train_test_split(features, labels,train_size=0.9,shuffle=True,random_state=42)
    # train_features和train_labels合并一个test,再保存到train中
    train_data = np.hstack((train_features,train_labels.reshape(-1,1)))
    np.savetxt('data/tforest/train/'+str(ord_num)+'_train.txt',train_data , delimiter=',', fmt='%.6f')
    # ver_features和ver
    ver_data = np.hstack((ver_features,ver_labels.reshape(-1,1)))
    np.savetxt('data/tforest/verification/' + str(ord_num) + '_ver.txt', ver_data, delimiter=',', fmt='%.6f')
def tforest_preprocessing():
    # 和stp_file_list 一样
    tf_file_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     13, 14, 15, 16, 17, 18, 19, 20,
                     22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                     35, 36, 37, 38, 39, 40, 41]
    for i in tf_file_list:
        tforest_data(i)


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

    tforest_preprocessing()
    # data_preprocessing()
    # stp_preprocessing()
    # mtp_preprocessing()
    # fcp_preprocessing()
    # test_data 就不用了 因为一开始已经处理好了，我只需要最后保存起来就ok
    # test_data()
