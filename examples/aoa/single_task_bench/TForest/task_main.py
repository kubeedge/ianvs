import logging
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import statsmodels.api as sm

from tools import data_io, load_test
from tree import Tree
from tqdm import tqdm
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

import scienceplots

plt.style.use(['science', 'ieee'])

logging.basicConfig(level=logging.INFO)


def tree_initial():
    tree_list = []
    all_train_features = []
    all_test_features = []
    all_train_labels = []
    all_test_labels = []
    for data_index in range(42):
        features, labels = data_io(data_index=data_index)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                        random_state=42, test_size=.2, shuffle=True)

        train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,
                                                        random_state=42, test_size=.2, shuffle=True)

        tree = Tree(train_features, val_features, train_labels, val_labels)
        tree.train()
        tree_list.append(tree)

        for train_feature in train_features:
            all_train_features.append(train_feature)
        for test_feature in test_features:
            all_test_features.append(test_feature)
        for train_label in train_labels:
            all_train_labels.append(train_label)
        for test_label in test_labels:
            all_test_labels.append(test_label)

    all_train_features = np.array(all_train_features)
    all_test_features = np.array(all_test_features)
    all_train_labels = np.array(all_train_labels)
    all_test_labels = np.array(all_test_labels)

    single_tree = Tree(all_train_features, all_test_features, all_train_labels, all_test_labels)
    single_tree.train()
    # tree_list.append(tree)

    return tree_list, single_tree


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
    for tree_index in range(1, len(tree_list)):
        print(f'\n与第{tree_index}个tree合并的情况')
        tree2 = tree_list[tree_index]
        tree_fused = Tree(np.concatenate((tree1.train_features, tree2.train_features), axis=0),
                            np.concatenate((tree1.test_features, tree2.test_features), axis=0),
                            np.concatenate((tree1.train_labels, tree2.train_labels), axis=0),
                            np.concatenate((tree1.test_labels, tree2.test_labels), axis=0))
        tree_fused.train()

        score1 = tree1.predict_score(tree_fused.model)
        score2 = tree2.predict_score(tree_fused.model)
        fusion_type = check_fusion_type(score1, score2, tree1, tree2)
        print(f'score1:{score1}, score2:{score2},合并类型: {fusion_type}')
        if best_tree is None:
            print('best_tree is None')
        else:

            print(f'tree_fused.r2:{tree_fused.r2},best_tree.r2:{best_tree.r2}')
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
        print(f'best_index为:{best_index}')
    logging.debug('Best Fusion: {} and {}'.format(str(fusion_type), str(best_index)))

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
        print(f'tree的数量为:{len(tree_list)}')
        for tree_each in tree_list:
            print(tree_each.r2,end = ',')
        tree_fused, fusion_flag, best_index = try_fusion(tree_list)
        if fusion_flag == 0:
            del tree_list[0]  # delete all two trees
            del tree_list[best_index-1]  # the index changed when the first on deleted.
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


def tforest_main():
    # logging.info("Tree Initializating")
    # tree_list, single_tree = tree_initial()
    #
    # joblib.dump(tree_list, 'initial_tree.joblib')
    # joblib.dump(single_tree, 'single_tree.joblib')
    tree_list = joblib.load('initial_tree.joblib')
    logging.info("Tree Updating")
    tree_list = tree_update(tree_list)
    for tree_each in tree_list:
        print(tree_each.r2)

    joblib.dump(tree_list, 'tree_final.joblib')


def cdf_draw_each(data, label, color='#8ea3c2', marker='*', linestyle='-', markersize=3):
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(min(data), max(data))
    y = ecdf(x)
    # plt.step(x, y, label=label, color=color, markersize=markersize, marker=marker, linestyle=linestyle)
    plt.step(x, y, label=label)


def model_selector_training(all_error_list, test_features):
    for index in range(8):
        error_list = np.array(all_error_list)[:, index]
        selector=AdaBoostRegressor(random_state=42)
        selector.fit(test_features, error_list)
        joblib.dump(selector, 'selectors/task-'+str(index)+'.m')


def task_allocation_comi(test_features, test_labels, single_tree):
    tree_final = joblib.load('tree_final.joblib')

    # model_list = []
    # for index, tree_each in enumerate(tree_final):

    #     # tree_each.train()
    #     # model_list.append(tree_each.model)
    #     print(index, tree_each.r2)
    #     model_list.append(tree_each.model)

    #     # joblib.dump(tree_each.model, 'models/task-'+str(index)+'.m')

    model_list = []
    for tree_index in range(len(tree_final)):
        model = joblib.load('models/task-'+str(tree_index)+'.joblib')
        model_list.append(model)

    selector_list = []
    for tree_index in range(len(tree_final)):
        selector = joblib.load('selectors/task-'+str(tree_index)+'.joblib')
        selector_list.append(selector)

    # selector = joblib.load('selector.joblib')
    
    mse_tomi = []
    mse_bsm = []
    ssm_list = []
    for test_index in tqdm(range(len(test_features))):
        features_each = test_features[test_index]
        labels_each = test_labels[test_index]

        best_task = 0
        min_error = 100

        for index, selector in enumerate(selector_list):
            predict_error = selector.predict([features_each])[0]
            # print(predict_error)
            if predict_error < min_error:
                min_error = predict_error
                best_task = index
        # print(min_error)
        # print()

        # best_task = selector.predict([features_each])[0]

        # print(best_task)
        prediction = model_list[best_task].predict([features_each])[0]
        # print(abs(prediction-labels_each))
        
        mse_tomi.append(abs(prediction-labels_each) * 180 / 3.14)

        ssm = abs(single_tree.predict([features_each])[0]-labels_each) * 180 / 3.14
        # print(ssm)
        ssm_list.append(ssm)
        # print()

        error_list = []
        for task_index in range(len(model_list)):
            prediction = model_list[task_index].predict([features_each])[0]
            error = abs(prediction-labels_each) * 180 / 3.14
            error_list.append(error)
        mse_bsm.append(min(error_list))

    cdf_draw_each(mse_tomi, 'COMI')
    cdf_draw_each(ssm_list, 'Single Model')
    cdf_draw_each(mse_bsm, 'Best Model Selection')
    plt.title('Mean error on Aoa')
    plt.legend()
    plt.savefig('cdf_comi.png')


def task_allocation_tforest(test_features, test_labels, single_tree):
    tree_final = joblib.load('tree_final.joblib')

    # model_list = []
    # for index, tree_each in enumerate(tree_final):

    #     # tree_each.train()
    #     # model_list.append(tree_each.model)
    #     print(index, tree_each.r2)
    #     model_list.append(tree_each.model)

    #     # joblib.dump(tree_each.model, 'models/task-'+str(index)+'.m')

    model_list = []
    for tree_index in range(len(tree_final)):
        model = joblib.load('models/task-'+str(tree_index)+'.joblib')
        model_list.append(model)

    # selector_list = []
    # for tree_index in range(len(tree_final)):
    #     selector = joblib.load('selectors/task-'+str(tree_index)+'.joblib')
    #     selector_list.append(selector)

    selector = joblib.load('selector.joblib')
    
    mse_tomi = []
    mse_bsm = []
    ssm_list = []
    for test_index in tqdm(range(len(test_features))):
        features_each = test_features[test_index]
        labels_each = test_labels[test_index]

        # best_task = 0
        # min_error = 100

        # for index, selector in enumerate(selector_list):
        #     predict_error = selector.predict([features_each])[0]
        #     # print(predict_error)
        #     if predict_error < min_error:
        #         min_error = predict_error
        #         best_task = index
        # # print(min_error)
        # # print()

        best_task = selector.predict([features_each])[0]

        # print(best_task)
        prediction = model_list[best_task].predict([features_each])[0]
        # print(abs(prediction-labels_each))
        
        mse_tomi.append(abs(prediction-labels_each) * 180 / 3.14)

        ssm = abs(single_tree.predict([features_each])[0]-labels_each) * 180 / 3.14
        # print(ssm)
        ssm_list.append(ssm)
        # print()

        error_list = []
        for task_index in range(len(model_list)):
            prediction = model_list[task_index].predict([features_each])[0]
            error = abs(prediction-labels_each) * 180 / 3.14
            error_list.append(error)
        mse_bsm.append(min(error_list))

    cdf_draw_each(mse_tomi, 'TForest')
    cdf_draw_each(ssm_list, 'Single Model')
    cdf_draw_each(mse_bsm, 'Best Model Selection')
    plt.title('Mean error on Aoa')
    plt.legend()
    plt.savefig('cdf_tforest.png')


def selector_training(error_list, test_features, index):
    error_list = np.array(error_list)
    selector=AdaBoostRegressor(random_state=42)
    selector.fit(test_features, error_list)
    joblib.dump(selector, 'selectors/task-'+str(index)+'.joblib')


def gen_error_list():
    single_tree = joblib.load('single_tree.joblib')
    train_features = single_tree.train_features
    train_labels = single_tree.train_labels

    tree_final = joblib.load('tree_final.joblib')

    model_list = []
    for tree_index in range(len(tree_final)):
        model = joblib.load('models/task-'+str(tree_index)+'.joblib')
        model_list.append(model)

    for index, model in enumerate(model_list):
        error_list = []
        # print(model.estimator_weights_)
        for sample_index in tqdm(range(len(train_features))):
            train_feature = train_features[sample_index]
            train_label = train_labels[sample_index]
            prediction = model.predict(train_feature.reshape(1, -1))[0]
            # print(prediction, train_label)
            error_list.append(abs(prediction - train_label))

        # print(error_list[0])
        selector_training(error_list, train_features, index)
        

def test_main():
    # load single task
    single_tree = joblib.load('single_tree.joblib')
    
    test_features = single_tree.test_features
    test_labels = single_tree.test_labels

    task_allocation_comi(test_features, test_labels, single_tree)


def save_model():
    tree_final = joblib.load('tree_final.joblib')
    for model_index in range(len(tree_final)):
        tree_final[model_index].train()
        print(tree_final[model_index].model.estimator_weights_)
        joblib.dump(tree_final[model_index].model, 'models/task-'+str(model_index)+'.joblib')


def task_selector():
    task_final = joblib.load('tree_final.joblib')
    features = []
    task_labels = []

    for index, task_each in enumerate(task_final):
        for feature_each in task_each.train_features:
            features.append(feature_each)
            task_labels.append(index)

    selector=AdaBoostClassifier(random_state=42)
    selector.fit(features, task_labels)
    joblib.dump(selector, 'selector.joblib')


if __name__ == '__main__':
    # tforest_main()
    # save_model()
    # gen_error_list()
    
    # task_selector()
    test_main()

    # def take_r2(elem):
    #     if elem.checked is True:
    #         return 1
    #     else:
    #         return elem.r2
    # # tforest_main()
    # final_tree = joblib.load('tree_final.joblib')
    #
    # init_tree = joblib.load('initial_tree.joblib')
    # init_tree.sort(key=take_r2,reverse = True)
    # SUM = 0
    # for i in range(len(final_tree)):
    #     if final_tree[i].r2 == init_tree[i].r2:
    #         SUM+=1
    #     print('final:',final_tree[i].r2)
    #     print('init:',init_tree[i].r2)
    # print('final:',len(final_tree))
    # print('init:',len(init_tree))
    # print('SUM:',SUM)
    # # print(len(tree_list))
    # print(1)

