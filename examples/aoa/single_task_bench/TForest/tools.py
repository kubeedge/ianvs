import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Base directory relative to this file
BASE_DIR = Path(__file__).resolve().parent


def data_io(data_index=1):
    data_dir = BASE_DIR / 'data' / 'processed'
    # names = [str(i) for i in range(64, 86)] #64~86
    # CHANNEL_NUM = 52
    names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
    file_path = data_dir / f'{names[data_index]}.txt'
    all_data = np.loadtxt(file_path, delimiter=',')
#     print(file_path)
# #     all_data = all_data.sample(frac=1.0)
#     print(all_data.shape)
    # snr = all_data[:, 6:10]
    all_features = all_data[:, 10:-1]

    # all_features = np.concatenate((snr, angles), axis = 1)
    # pl = PolynomialFeatures()
    # all_features = pl.fit_transform(all_features)
    all_labels = all_data[:, 5]

    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
    #                                                 random_state=42, test_size=.01, shuffle=False)

    # print(features.shape)

    # regr = AdaBoostRegressor()
    # regr.fit(train_features, train_labels)
    # print(regr.score(test_features, test_labels))

    # test_predict = regr.predict(test_features)
    # plt.plot(test_predict, label = 'test_predict')
    # plt.plot(test_labels, label = 'test_labels')
    # plt.legend()
    # plt.savefig('test.png')

    return all_features, all_labels


def load_test(test_file=None):
    if test_file is None:
        test_file = BASE_DIR / "seq" / "seq83_new.txt"
    else:
        test_file = Path(test_file)
        if not test_file.is_absolute():
            test_file = BASE_DIR / test_file
    CHANNEL_NUM = 52
    test_seq = np.array(pd.read_table(test_file, header = None, sep=','))
    snr = test_seq[:, 6:10]
    angles = test_seq[:, 10:]
    angles = np.concatenate((angles[:, 0:CHANNEL_NUM] - angles[:, CHANNEL_NUM:2*CHANNEL_NUM],
                    angles[:, 0:CHANNEL_NUM] - angles[:, 2*CHANNEL_NUM:3*CHANNEL_NUM],
                    angles[:, :CHANNEL_NUM] - angles[:, 3*CHANNEL_NUM:4*CHANNEL_NUM],
                    angles[:, CHANNEL_NUM:2*CHANNEL_NUM] - angles[:, 2*CHANNEL_NUM:3*CHANNEL_NUM],
                    angles[:, CHANNEL_NUM:2*CHANNEL_NUM] - angles[:, 3*CHANNEL_NUM:4*CHANNEL_NUM],
                    angles[:, 2*CHANNEL_NUM:3*CHANNEL_NUM] - angles[:, 3*CHANNEL_NUM:4*CHANNEL_NUM]),
                    axis = 1)

    all_features = np.concatenate((snr, angles), axis = 1)
    # pl = PolynomialFeatures()
    # all_features = pl.fit_transform(all_features)
    all_labels = test_seq[:, 5]
    return all_features, all_labels


if __name__ == '__main__':
    features, labels = load_test()
    print(features.shape)
