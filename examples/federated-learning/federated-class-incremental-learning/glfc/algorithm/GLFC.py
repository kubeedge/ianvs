import copy
import numpy as np
import tensorflow as tf
import keras
import logging 
from network import NetWork, incremental_learning, copy_model

def get_one_hot(target, num_classes):
    # print(f'in get  one hot, target shape is {target.shape}')
    y = tf.one_hot(target, depth=num_classes)
    # print(f'in get  one hot, y shape is {y.shape}')
    if len(y.shape) == 3:
        y = tf.squeeze(y, axis=1)
    # print(f'in get  one hot, after tf.squeeze y shape is {y.shape}')
    return y

class GLFC_Client:
    def __init__(self, feature_extractor, num_classes, batch_size, task_size, memory_size, epochs, learning_rate, encode_model):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = NetWork(num_classes, feature_extractor)
        self.encode_model = encode_model
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.task_size = task_size
        
        self.old_model = None 
        self.train_set = None 
        
        self.exemplar_set = [] # 
        self.class_mean_set = []
        self.learned_classes = []
        self.learned_classes_numebr = 0
        self.memory_size = memory_size
        
        self.old_task_id = -1
        self.current_class = None
        self.last_class = None 
        self.train_loader = None 
        
    def before_train(self, task_id, train_data, class_learned, old_model):
        logging.info(f"------before train task_id: {task_id}------")
        # print(f'train data len is :{len(train_data[1])}')
        need_update = (task_id != self.old_task_id)
        if need_update:
            self.old_task_id = task_id
            self.num_classes = self.task_size * (task_id + 1)
            if self.current_class is not None: 
                self.last_class = self.current_class
            logging.info(f'self.last_class is , {self.last_class}, {self.num_classes}')
            self.model = incremental_learning(self.model, self.num_classes)
            self.current_class = np.unique(train_data[1]).tolist()
            self.update_new_set(need_update)
            if len(old_model) != 0:
                self.old_model = old_model[1]
            # incremental_learning(self.model, self.num_classes * (task_id+ 1))
        else:
            if len(old_model) != 0:
                self.old_model = old_model[0]
        self.train_set = train_data
        self.train_loader = self._get_train_loader(True)
        logging.info(f'------finish before train task_id: {task_id}------')
    
    def update_new_set(self, need_update):
        if need_update and self.last_class is not None:
            # update exemplar
            self.learned_classes += self.last_class
            self.learned_classes_numebr += len(self.last_class)
            m = int(self.memory_size  / self.learned_classes_numebr)
            self._reduce_exemplar_set(m)
            for i in self.last_class:
                images = self.get_train_set_data(i)
                # print(f'process class {i} with {len(images)} images')
                self._construct_exemplar_set(images, i, m)
        # print(f'-------------Learned classes: {self.learned_classes} current classes :{self.current_class} last classes : {self.last_class}--------------')
        
            
            
    def _get_train_loader(self, mix):
        # print(self.train_set[0].shape, self.train_set[1].shape)
        if mix :
            for exm_set in self.exemplar_set:
                logging.info(f'mix the exemplar{len(exm_set[0])}, {len(exm_set[1])}')
                label = np.array(exm_set[1])
                label = label.reshape(-1, 1)
                self.train_set[0] = np.concatenate((self.train_set[0],exm_set[0]), axis=0)
                self.train_set[1] = np.concatenate((self.train_set[1],label), axis=0)
        logging.info(f'{self.train_set[0].shape}, {self.train_set[1].shape}')
        return tf.data.Dataset.from_tensor_slices((self.train_set[0], self.train_set[1])).shuffle(buffer_size=10000000).batch(self.batch_size)
    
    def train(self, round):
        opt = keras.optimizers.SGD(learning_rate=self.learning_rate, weight_decay=0.00001)
        # print(self.train_loader is None)
        # self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        for epoch in range(self.epochs):
            for step, (x, y) in enumerate(self.train_loader):
                # opt = keras.optimizers.SGD(learning_rate=self.learning_rate, weight_decay=0.00001)
                # self.model.fit(x, y, epochs=3)
                with tf.GradientTape() as tape:
                    # logits = self.model(x, training=True)
                    # y = get_one_hot(y, self.num_classes)
                    # loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, logits, from_logits=True))
                    loss = self._compute_loss(x, y)
                logging.info(f'------round{round} epoch{epoch} step{step} loss: {loss} and loss dim is {loss.shape}------')
                grads = tape.gradient(loss, self.model.trainable_variables)
                # print(f'grads shape is {len(grads)} and type is {type(grads)}')
                opt.apply_gradients(zip(grads, self.model.trainable_variables))
        
        logging.info(f'------finish round{round} traning------')
    
    def _compute_loss(self, imgs, labels):
        logging.info(f'self.old_model is available: {self.old_model is not None}')
        y_pred = self.model(imgs, training=True)
        target = get_one_hot(labels, self.num_classes)
        logits = y_pred
            # prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(logits, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        pred = tf.reshape(pred, labels.shape)
        
        y = tf.cast(labels, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        logging.info(f'correct is {correct} and acc is {correct/imgs.shape[0]}')
            # print(f"total_correct: {total_correct}, total_num: {total_num}")
        if self.old_model == None:
            w = self.efficient_old_class_weight(target, labels )
            # print(f"old class weight shape: {w.shape}")
            loss = tf.reduce_mean(keras.losses.binary_crossentropy(target, y_pred, from_logits=True) * w)
            # print(f'in _compute_loss, without old model loss is {loss} and shape is {loss.shape}')
            return loss
        else :
            w = self.efficient_old_class_weight(target, labels)
            # print(f"old class weight shape: {w.shape}")
            loss = tf.reduce_mean(keras.losses.binary_crossentropy(target, y_pred, from_logits=True) * w)
            # print(f'in _compute_loss, loss is {loss} and shape is {loss.shape}')
            distill_target = tf.Variable(get_one_hot(labels, self.num_classes))
            # print(f"distill_target shape: {distill_target.shape} type: {type(distill_target)}")
            old_target = tf.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            # print(f'old_target shape: {old_target.shape} and old_task_size: {old_task_size}')
            distill_target[:, :old_task_size].assign(old_target)
            loss_old = tf.reduce_mean(keras.losses.binary_crossentropy(distill_target, y_pred, from_logits=True))
            logging.info(f'loss old shape is {loss_old.shape}')
            return 0.5 * loss + 0.5 * loss_old
        
    def efficient_old_class_weight(self, output, labels):
        # print("---calculate efficient old class weight---")
        pred = tf.sigmoid(output)
        # print(f"labels.shape : {labels.shape}")
        N, C = pred.shape
        # print(f"pred shape: {pred.shape}")
        class_mask = tf.zeros([N, C], dtype=tf.float32)
        class_mask = tf.Variable(class_mask)
        # print(f"class_mask shape: {class_mask.shape}")
        ids = np.zeros([N,2], dtype=np.int32)
        for i in range(N):
            ids[i][0] = i
            ids[i][1] = labels[i]
        # print(f"ids shape: {ids.shape}")
        updates = tf.ones([N], dtype=tf.float32)
        # print(f"updates shape: {updates.shape}")
        class_mask = tf.tensor_scatter_nd_update(class_mask, ids, updates)
        # print(f"class_mask shape: {class_mask.shape}")
        target = get_one_hot(labels, self.num_classes)
        # print(f'target shape: {target.shape}')
        g = tf.abs(target - pred)
        g = tf.reduce_sum(g*class_mask, axis=1)
        # print(f"g shape: {g.shape}")
        idx = tf.cast(tf.reshape( labels ,(-1,1)), tf.int32)
        if len(self.learned_classes) != 0:
            learned_classes_tensor = tf.constant(self.learned_classes, dtype=tf.int32)
            for i in self.learned_classes:
                mask = tf.math.not_equal(idx, i)
                negative_value = tf.cast(tf.fill(tf.shape(idx), -1), tf.int32)
                idx = tf.where(mask, idx, negative_value)

             # 计算 index1 和 index2
            index1 = tf.cast(tf.equal(idx, -1), tf.float32)
            index2 = tf.cast(tf.not_equal(idx, -1), tf.float32)

            # 计算 w1 和 w2
            w1 = tf.where(
                tf.not_equal(tf.reduce_sum(index1), 0),
                tf.math.divide(g * index1, (tf.reduce_sum(g * index1) / tf.reduce_sum(index1))),
                tf.zeros_like(g)
            )
            # print(f"w1 shape: {w1.shape}")
            w2 = tf.where(
                tf.not_equal(tf.reduce_sum(index2), 0),
                tf.math.divide(g * index2, (tf.reduce_sum(g * index2) / tf.reduce_sum(index2))),
                tf.zeros_like(g)
            )
            # print(f"w2 shape: {w2.shape}")
            # 计算最终的 w
            w = w1 + w2
            return w
        else:
            return tf.ones(g.shape, dtype=tf.float32)
        
    def get_train_set_data(self, class_id):
        images = []
        # print(len(self.train_set[0]))
        for i in range(len(self.train_set[0])):
            # print(self.train_set[1][i])
            if self.train_set[1][i] == class_id:
                # print(self.train_set[0][i].shape)
                images.append(self.train_set[0][i])
        return images
    
    def _reduce_exemplar_set(self, m):
        for i in range(len(self.exemplar_set)):
            old_exemplar_data = self.exemplar_set[i][0][:m]
            old_exemplar_label = self.exemplar_set[i][1][:m]
            self.exemplar_set[i] = (old_exemplar_data, old_exemplar_label)
            
    def _construct_exemplar_set(self, images,label,  m):
        class_mean, fe_outpu = self.compute_class_mean(images)
        exemplar = []
        labels = []
        now_class_mean = np.zeros((1,512))
        for i in range(m):
            x = class_mean - (now_class_mean + fe_outpu)/(i + 1)
            x = np.linalg.norm(x)
            index = np.argmin(x)
            now_class_mean += fe_outpu[index]
            # print(
            #     'construct exemplar: ',len(images[index]),
            #     'type: ',type(images[index]),
            #     'shape: ',images[index].shape
            #     )
            exemplar.append(images[index])
            labels.append(label)
        self.exemplar_set.append((exemplar, labels))
        
    def compute_class_mean(self, images):
        images_data = tf.data.Dataset.from_tensor_slices(images).batch(self.batch_size)
        fe_output = self.model.feature_extractor(inputs=images_data)
        fe_output = tf.nn.l2_normalize( fe_output).numpy()
        # print(f"fe_output shape is {fe_output.shape}")
        class_mean = tf.reduce_mean(fe_output, axis=0)
        # print(f'class mean is {class_mean.shape}')
        return class_mean, fe_output
    
    def proto_grad(self):
        iters = 50
        cri_loss = keras.losses.SparseCategoricalCrossentropy()
        proto = []
        proto_grad = []
        for i in self.current_class:
            images = self.get_train_set_data(i)
            # print(f'image shape is {len(images)}')
            class_mean, fe_output = self.compute_class_mean(images)
            dis = np.linalg.norm(class_mean - fe_output, axis=1)
            pro_index = np.argmin(dis)
            proto.append(images[pro_index])
            
        for i in range(len(proto)):
            data = proto[i]
            data = tf.cast(tf.expand_dims(data, axis=0), tf.float32)
            # print(f"in proto_grad, data shape is {data.shape}")
            label = self.current_class[i]
            # print("in proto_grad, label shape is ", label.shape)
            label = tf.constant([label])
            target = get_one_hot(label, self.num_classes)
            proto_model = copy_model(self.model)
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, weight_decay=0.00001)
            for _ in range(iters):
                with tf.GradientTape() as tape:
                    output = proto_model(data)
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(target, output))
                grads = tape.gradient(loss, proto_model.trainable_variables)
                opt.apply_gradients(zip(grads, proto_model.trainable_variables))
            with tf.GradientTape() as tape:
                outputs = self.encode_model(data)
                loss_cls = cri_loss(label, outputs)
            dy_dx = tape.gradient(loss_cls, self.encode_model.trainable_variables)
            # print(f"dy_dx shape is {len(dy_dx)} and type is {type(dy_dx)}")
            original_dy_dx = [tf.identity(grad) for grad in dy_dx]
            proto_grad.append(original_dy_dx)
        return proto_grad
    
    
    def evaluate(self):
        logging.info('evaluate')
        total_num = 0
        total_correct = 0
        for x, y in self.train_loader:
            logits = self.model(x, training=False)
            # prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            pred = tf.reshape(pred, y.shape)
            print(pred)
            y = tf.cast(y, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
            # print(f"total_correct: {total_correct}, total_num: {total_num}")
        acc = total_correct / total_num
        del total_correct
        logging.info(f"finsih task {self.old_task_id} evaluate, acc: {acc}")
        return acc