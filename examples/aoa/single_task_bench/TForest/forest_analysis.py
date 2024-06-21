import joblib
from tree import Tree


def forest_number():
    tree_final = joblib.load('tree_final.joblib')

    for tree_index in range(len(tree_final)):
        print(len(tree_final[tree_index].train_features))


if __name__ == '__main__':
    forest_number()
    