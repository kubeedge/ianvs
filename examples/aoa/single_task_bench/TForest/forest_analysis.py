<<<<<<< HEAD
import joblib
from tree import Tree


def forest_number():
    tree_final = joblib.load('tree_final.joblib')

    for tree_index in range(len(tree_final)):
        print(len(tree_final[tree_index].train_features))


if __name__ == '__main__':
    forest_number()
    
=======
version https://git-lfs.github.com/spec/v1
oid sha256:432e739524681904042d3d4a375b0c5d701e2ae9b3df7440abdd972bf8421d66
size 267
>>>>>>> 9676c3e (ya toh aar ya toh par)
