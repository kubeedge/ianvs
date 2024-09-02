import tensorflow as tf
import numpy as np
import os


def process_cifar100():
    if not os.path.exists('/home/wyd/ianvs/project/data/cifar100'):
        os.makedirs('/home/wyd/ianvs/project/data/cifar100')
    train_txt = '/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt'
    with open(train_txt, 'w') as f:
        pass
    test_txt = '/home/wyd/ianvs/project/data/cifar100/cifar100_test.txt'
    with open(test_txt, 'w') as f:
        pass
    # 加载CIFAR-100数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    print(y_test.shape)
    # 数据预处理：归一化
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 将标签转换为类别索引
    class_labels = np.unique(y_train)  # 获取所有类别
    train_class_dict = {label: [] for label in class_labels}
    test_class_dict = {label: [] for label in class_labels}
    # train_cnt = 0
    # train_file_str = []
    # 按类别组织训练数据
    for img, label in zip(x_train, y_train):
        # print(type(img))
        # print('----')
        train_class_dict[label[0]].append(img)
    #     np.save(f'../../../../../data/cifar100/cifar100_train_index_{train_cnt}.npy', img)
    #     train_file_str.append(f'cifar100_train_index_{train_cnt}.npy\t{label}\n')
    #     train_cnt += 1
    # test_cnt = 0
    # test_file_str = []
    # # 按类别组织测试数据
    for img, label in zip(x_test, y_test):
        # test_class_dict[label[0]].append(img)
        test_class_dict[label[0]].append(img)
    #     np.save(f'../../../../../data/cifar100/cifar100_test_index_{test_cnt}.npy', img)
    #     test_file_str.append(f'cifar100_train_index_{test_cnt}.npy\t{label[0]}\n')
    #     test_cnt += 1
    # for line in train_file_str:
    #     with open(train_txt, 'a') as f:
    #         f.write(line)
    # for line in test_file_str:
    #     with open(test_txt, 'a') as f:
    #         f.write(line)

    # 保存训练数据到本地文件
    for label, imgs in train_class_dict.items():
        data = np.array(imgs)
        print(data.shape)
        np.save(f'/home/wyd/ianvs/project/data/cifar100/cifar100_train_index_{label}.npy',data)
        with open(train_txt, 'a') as f:
            f.write(f'/home/wyd/ianvs/project/data/cifar100/cifar100_train_index_{label}.npy\t{label}\n')
    # 保存测试数据到本地文件
    for label, imgs in test_class_dict.items():
        np.save(f'/home/wyd/ianvs/project/data/cifar100/cifar100_test_index_{label}.npy', np.array(imgs))
        with open(test_txt, 'a') as f:
            f.write(f'/home/wyd/ianvs/project/data/cifar100/cifar100_test_index_{label}.npy\t{label}\n')
    print(f'CIFAR-100 数据集已按类别保存到本地文件。')



if __name__ == '__main__':
    process_cifar100()
    # arr = np.load("/home/wyd/ianvs/project/data/cifar100/cifar100_train_index_0.npy")
    # print(arr.shape)
    # print(arr)
    # process_cifar100()
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    # mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
    # std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
    # x_train = x_train[:5000]
    # y_train = y_train[:5000]
    # batch_size=32
    # train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).map(
    #         lambda x, y: (
    #             (tf.cast(x, dtype=tf.float32) / 255. - mean) / std,
    #             tf.cast(y, dtype=tf.int32)
    #         )
    #     )
    # from algorithm.resnet import resnet18
    # model = resnet18(100)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    # for epoch in range(10):
    #     for _, (x,y) in enumerate(train_db):
    #         with tf.GradientTape() as tape:
    #             logits = model(x, training=True)
    #             y = tf.one_hot(y, depth=100)
    #             y = tf.squeeze(y, axis=1)
    #             loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))
    #         grads = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #         print(f"train round {1}: Epoch {epoch + 1} loss: {loss.numpy():.4f}")
    # total_num = 0
    # total_correct = 0
    # for _, (x,y) in enumerate(train_db):
    #     logits = model(x, training=False)
    #     # prob = tf.nn.softmax(logits, axis=1)
    #     pred = tf.argmax(logits, axis=1)
    #     pred = tf.cast(pred, dtype=tf.int32)
    #     pred = tf.reshape(pred, y.shape)
    #     # print(pred.shape, y.shape)
    #     correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
    #     correct = tf.reduce_sum(correct)
    #
    #     total_num += x.shape[0]
    #     total_correct += int(correct)
    #     print(f"total_correct: {total_correct}, total_num: {total_num}")
    # acc = total_correct / total_num
    # del total_correct
    # print(f"finsih round {round}evaluate, acc: {acc}")
