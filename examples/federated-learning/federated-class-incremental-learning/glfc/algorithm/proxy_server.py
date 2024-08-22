import keras
import copy
import numpy as np
import tensorflow as tf
import logging 
from network import * 
logging.getLogger().setLevel(logging.INFO)

class ProxyData:
    def __init__(self):
        self.test_data = []
        self.test_label = []

class ProxyServer:
    def __init__(self, learning_rate, num_class, feature_extractor, encode_model,  **kwargs):
        self.learning_rate = learning_rate
        self.feature_extractor = feature_extractor
        self.encode_model = encode_model

        self.model = NetWork(num_class, feature_extractor)
        self.model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(np.random.rand(1, 32, 32, 3), np.random.randint(0, num_class, 1), epochs=1)
        
        self.monitor_dataset = ProxyData() 
        self.new_set =[]
        self.new_set_label = []
        self.num_classes= 0
        self.proto_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0
        
        self.num_image = 20
        self.Iteration = 250
        
    
    def model_back(self):
        return [self.best_model_1, self.best_model_2]
    
    def dataload(self, proto_grad):
        self.proto_grad = proto_grad
        if len(proto_grad )  != 0 :
            self.reconstruction()
            self.monitor_dataset.test_data = self.new_set
            self.monitor_dataset.test_label = self.new_set_label
            self.last_perf = 0
            self.best_model_1 = self.best_model_2
        cur_perf = self.monitor()
        logging.info(f'in proxy server, current performance is {cur_perf}')
        if cur_perf > self.best_perf:
            self.best_perf = cur_perf
            self.best_model_2 = copy_model(self.model)
            
    
    def monitor(self):
        correct, total = 0, 0
        for (x, y) in zip(self.monitor_dataset.test_data, self.monitor_dataset.test_label):
            y_pred = self.model(x)
            
            predicts = tf.argmax(y_pred, axis=-1)
            predicts = tf.cast(predicts, tf.int32)
            logging.info(f'y_pred shape {predicts} and y {y}')
            correct += tf.reduce_sum( tf.cast(tf.equal(predicts, y), dtype=tf.int32))
            total += x.shape[0]
        acc = 100 * correct / total
        return acc
    
    def grad2label(self):
        # print("-------------------grad2label-------------------")
        proto_grad_label = []
        for w_single in self.proto_grad:
            # 计算 w_single[-2] 的 sum 并找到其最小值的索引
            # print(w_single.shape)
            # print(w_single[-2].shape)
            # print(w_single[-2])
            # print(tf.reduce_sum(w_single[-2], axis=-1).shape)
            # print(tf.reduce_sum(w_single[-2], axis=-1))
            pred = tf.argmin(tf.reduce_sum(w_single[-2], axis=-1), axis=-1)
            proto_grad_label.append(pred)
        return proto_grad_label
    
    
    def reconstruction(self):
        self.new_set = []
        self.new_set_label = []
        proto_label = self.grad2label()
        proto_label = np.array(proto_label)
        # print(f"pooling label: {proto_label}")
        class_ratio  = np.zeros((1, 100))
        
        for i in proto_label:
            class_ratio[0][i] += 1
        
        for label_i in range(100):
            if class_ratio[0][label_i] > 0 :
                # num_agumentation = self.num_image
                agumentation = []
                
                grad_index = np.where(proto_label == label_i)
                logging.info(f'grad index : {grad_index} and label is {label_i}')
                for j in range(len(grad_index[0])):
                    grad_true_temp = self.proto_grad[grad_index[0][j]]
                    
                    dummy_data = tf.Variable(np.random.rand(1, 32, 32, 3), trainable=True)
                    # print("dummy_data", dummy_data._unique_id)
                    label_pred = tf.constant([label_i])
                    
                    opt = keras.optimizers.SGD(learning_rate=0.1)
                    cri = keras.losses.SparseCategoricalCrossentropy()
                    
                    recon_model = copy.deepcopy(self.encode_model)
                    
                    for iter in range(self.Iteration):
                        with tf.GradientTape() as tape0:
                            with tf.GradientTape() as tape1:
                                y_pred = recon_model(dummy_data)
                                loss = cri(label_pred, y_pred)
                            # print(f'iter {iter} loss {loss} and loss shape {loss.shape}')
                            dummy_dy_dx = tape1.gradient(loss, recon_model.trainable_variables)
                        
                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad_true_temp):
                                # print(gx.shape, gy.shape)
                                gx = tf.cast(gx, tf.double)
                                gy = tf.cast(gy, tf.double)
                                sub_value = tf.subtract(gx, gy)
                                pow_value = tf.pow(sub_value, 2)
                                grad_diff += tf.reduce_sum(pow_value)
                            # print(f'grad_diff {grad_diff} and grad_diff shape {grad_diff.shape}')
                        grad = tape0.gradient(grad_diff, dummy_data)
                        # print(f' grad_diff shape {grad.shape} and type(grad) {type(grad)}')
                        opt.apply_gradients(zip([grad], [dummy_data]))
                        
                        # if iter == self.Iteration - 1:
                        #     print(f'iter {iter} loss {loss}')
                        
                        if iter >= self.Iteration - self.num_image:
                            # print(type(dummy_data))
                            dummy_data_temp = np.asarray(dummy_data)
                            agumentation.append(dummy_data_temp)
                            
                self.new_set.extend(agumentation)
                # print(len(agumentation))
                self.new_set_label.extend([label_i])
                        