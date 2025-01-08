# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import os
import tempfile
import time
import zipfile
import cv2
import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from FPN_TensorFlow.help_utils.help_utils import draw_box_cv
from FPN_TensorFlow.libs.label_name_dict.label_dict import NAME_LABEL_MAP
from FPN_TensorFlow.data.io.read_tfrecord import next_batch_for_tasks, convert_labels
from FPN_TensorFlow.data.io import image_preprocess
from FPN_TensorFlow.help_utils.tools import mkdir, view_bar, get_single_label_dict, single_label_eval
from FPN_TensorFlow.libs.configs import cfgs
from FPN_TensorFlow.libs.box_utils.show_box_in_tensor import draw_box_with_color, draw_boxes_with_categories
from FPN_TensorFlow.libs.fast_rcnn import build_fast_rcnn
from FPN_TensorFlow.libs.networks.network_factory import get_flags_byname, get_network_byname
from FPN_TensorFlow.libs.rpn import build_rpn

FLAGS = get_flags_byname(cfgs.NET_NAME)

# avoid the conflict: 1. tf parses flags with sys.argv; 2. test system also parses flags .
tf.flags.DEFINE_string("benchmarking_config_file", "", "ignore")

# close global warning log
# reason: during the running of tensorflow, a large number of warning logs will be printed
#         and these will submerge some important logs and increase inference latency.
# After disable the global warning job, that will not affect the running of application.
# if you want to open the global warning log, please comment(e.g: #) the statement.
# todo: 1. disable the local warning log instead of the global warning log.
#          e.g.: only to disable tensorflow warning log.

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

# set backend
os.environ['BACKEND_TYPE'] = 'TENSORFLOW'


@ClassFactory.register(ClassType.GENERAL, alias="FPN")
class BaseModel:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """

        self.has_fast_rcnn_predict = False

        self._init_tf_graph()

        self.temp_dir = tempfile.mkdtemp()
        if not os.path.isdir(self.temp_dir):
            mkdir(self.temp_dir)

        os.environ["MODEL_NAME"] = "model.zip"
        cfgs.LR = kwargs.get("learning_rate", 0.0001)
        cfgs.MOMENTUM = kwargs.get("momentum", 0.9)
        cfgs.MAX_ITERATION = kwargs.get("max_iteration", 5)

    def train(self, train_data, valid_data=None, **kwargs):

        if train_data is None or train_data.x is None or train_data.y is None:
            raise Exception("Train data is None.")

        with tf.Graph().as_default():

            img_name_batch, train_data, gtboxes_and_label_batch, num_objects_batch, data_num = \
                next_batch_for_tasks(
                    (train_data.x, train_data.y),
                    dataset_name=cfgs.DATASET_NAME,
                    batch_size=cfgs.BATCH_SIZE,
                    shortside_len=cfgs.SHORT_SIDE_LEN,
                    is_training=True,
                    save_name="train"
                )

            with tf.name_scope('draw_gtboxes'):
                gtboxes_in_img = draw_box_with_color(train_data, tf.reshape(gtboxes_and_label_batch, [-1, 5])[:, :-1],
                                                     text=tf.shape(gtboxes_and_label_batch)[1])

            # ***********************************************************************************************
            # *                                         share net                                           *
            # ***********************************************************************************************
            _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                              inputs=train_data,
                                              num_classes=None,
                                              is_training=True,
                                              output_stride=None,
                                              global_pool=False,
                                              spatial_squeeze=False)

            # ***********************************************************************************************
            # *                                            rpn                                              *
            # ***********************************************************************************************
            rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                                inputs=train_data,
                                gtboxes_and_label=tf.squeeze(gtboxes_and_label_batch, 0),
                                is_training=True,
                                share_head=cfgs.SHARE_HEAD,
                                share_net=share_net,
                                stride=cfgs.STRIDE,
                                anchor_ratios=cfgs.ANCHOR_RATIOS,
                                anchor_scales=cfgs.ANCHOR_SCALES,
                                scale_factors=cfgs.SCALE_FACTORS,
                                base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                                level=cfgs.LEVEL,
                                top_k_nms=cfgs.RPN_TOP_K_NMS,
                                rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                                max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                                rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                                # iou>=0.7 is positive box, iou< 0.3 is negative
                                rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                                rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                                rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                                remove_outside_anchors=False,  # whether remove anchors outside
                                rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

            rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

            rpn_location_loss, rpn_classification_loss = rpn.rpn_losses()
            rpn_total_loss = rpn_classification_loss + rpn_location_loss

            with tf.name_scope('draw_proposals'):
                # score > 0.5 is object
                rpn_object_boxes_indices = tf.reshape(tf.where(tf.greater(rpn_proposals_scores, 0.5)), [-1])
                rpn_object_boxes = tf.gather(rpn_proposals_boxes, rpn_object_boxes_indices)

                rpn_proposals_objcet_boxes_in_img = draw_box_with_color(train_data, rpn_object_boxes,
                                                                        text=tf.shape(rpn_object_boxes)[0])
                rpn_proposals_boxes_in_img = draw_box_with_color(train_data, rpn_proposals_boxes,
                                                                 text=tf.shape(rpn_proposals_boxes)[0])
            # ***********************************************************************************************
            # *                                         Fast RCNN                                           *
            # ***********************************************************************************************

            fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=train_data,
                                                 feature_pyramid=rpn.feature_pyramid,
                                                 rpn_proposals_boxes=rpn_proposals_boxes,
                                                 rpn_proposals_scores=rpn_proposals_scores,
                                                 img_shape=tf.shape(train_data),
                                                 roi_size=cfgs.ROI_SIZE,
                                                 roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                                 scale_factors=cfgs.SCALE_FACTORS,
                                                 gtboxes_and_label=tf.squeeze(gtboxes_and_label_batch, 0),
                                                 fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                                 fast_rcnn_maximum_boxes_per_img=100,
                                                 fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                                 show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                                 # show detections which score >= 0.6
                                                 num_classes=cfgs.CLASS_NUM,
                                                 fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                                 fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                                 # iou>0.5 is positive, iou<0.5 is negative
                                                 fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                                 use_dropout=False,
                                                 weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                                 is_training=True,
                                                 level=cfgs.LEVEL)

            fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
                fast_rcnn.fast_rcnn_predict()
            fast_rcnn_location_loss, fast_rcnn_classification_loss = fast_rcnn.fast_rcnn_loss()
            fast_rcnn_total_loss = fast_rcnn_location_loss + fast_rcnn_classification_loss

            with tf.name_scope('draw_boxes_with_categories'):
                fast_rcnn_predict_boxes_in_imgs = draw_boxes_with_categories(img_batch=train_data,
                                                                             boxes=fast_rcnn_decode_boxes,
                                                                             labels=detection_category,
                                                                             scores=fast_rcnn_score)

            # train
            added_loss = rpn_total_loss + fast_rcnn_total_loss
            total_loss = tf.losses.get_total_loss()

            global_step = tf.train.get_or_create_global_step()

            lr = tf.train.piecewise_constant(global_step,
                                             boundaries=[np.int64(20000), np.int64(40000)],
                                             values=[cfgs.LR, cfgs.LR / 10, cfgs.LR / 100])
            tf.summary.scalar('lr', lr)
            optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)

            train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)  # rpn_total_loss,
            # train_op = optimizer.minimize(second_classification_loss, global_step)

            # ***********************************************************************************************
            # *                                          Summary                                            *
            # ***********************************************************************************************
            # ground truth and predict
            tf.summary.image('img/gtboxes', gtboxes_in_img)
            tf.summary.image('img/faster_rcnn_predict', fast_rcnn_predict_boxes_in_imgs)
            # rpn loss and image
            tf.summary.scalar('rpn/rpn_location_loss', rpn_location_loss)
            tf.summary.scalar('rpn/rpn_classification_loss', rpn_classification_loss)
            tf.summary.scalar('rpn/rpn_total_loss', rpn_total_loss)

            tf.summary.scalar('fast_rcnn/fast_rcnn_location_loss', fast_rcnn_location_loss)
            tf.summary.scalar('fast_rcnn/fast_rcnn_classification_loss', fast_rcnn_classification_loss)
            tf.summary.scalar('fast_rcnn/fast_rcnn_total_loss', fast_rcnn_total_loss)

            tf.summary.scalar('loss/added_loss', added_loss)
            tf.summary.scalar('loss/total_loss', total_loss)

            tf.summary.image('rpn/rpn_all_boxes', rpn_proposals_boxes_in_img)
            tf.summary.image('rpn/rpn_object_boxes', rpn_proposals_objcet_boxes_in_img)
            # learning_rate
            tf.summary.scalar('learning_rate', lr)

            summary_op = tf.summary.merge_all()
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            restorer = self._get_restorer()
            saver = tf.train.Saver(max_to_keep=3)
            self.checkpoint_path = self.load(Context.get_parameters("base_model_url"))

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = False
            with tf.Session(config=config) as sess:
                sess.run(init_op)
                if self.checkpoint_path:
                    restorer.restore(sess, self.checkpoint_path)
                    print('restore model')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)

                summary_path = os.path.join(self.temp_dir, 'output/{}'.format(cfgs.DATASET_NAME),
                                            FLAGS.summary_path, cfgs.VERSION)

                mkdir(summary_path)
                summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

                for step in range(cfgs.MAX_ITERATION):
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    start = time.time()

                    _global_step, _img_name_batch, _rpn_location_loss, _rpn_classification_loss, \
                    _rpn_total_loss, _fast_rcnn_location_loss, _fast_rcnn_classification_loss, \
                    _fast_rcnn_total_loss, _added_loss, _total_loss, _ = \
                        sess.run([global_step, img_name_batch, rpn_location_loss, rpn_classification_loss,
                                  rpn_total_loss, fast_rcnn_location_loss, fast_rcnn_classification_loss,
                                  fast_rcnn_total_loss, added_loss, total_loss, train_op])

                    end = time.time()

                    if step % 50 == 0:
                        print("""{}: step{} image_name:{}
                             rpn_loc_loss:{:.4f} | rpn_cla_loss:{:.4f} | rpn_total_loss:{:.4f}
                             fast_rcnn_loc_loss:{:.4f} | fast_rcnn_cla_loss:{:.4f} | fast_rcnn_total_loss:{:.4f}
                             added_loss:{:.4f} | total_loss:{:.4f} | pre_cost_time:{:.4f}s"""
                              .format(training_time, _global_step, str(_img_name_batch[0]), _rpn_location_loss,
                                      _rpn_classification_loss, _rpn_total_loss, _fast_rcnn_location_loss,
                                      _fast_rcnn_classification_loss, _fast_rcnn_total_loss, _added_loss, _total_loss,
                                      (end - start)))

                    if step % 500 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, _global_step)
                        summary_writer.flush()

                    if step > 0 and step == cfgs.MAX_ITERATION - 1:
                        self.checkpoint_path = os.path.join(self.temp_dir, '{}_'.format(
                            cfgs.DATASET_NAME) + str(_global_step) + "_" + str(time.time()) + '_model.ckpt')
                        saver.save(sess, self.checkpoint_path)
                        print('Weights have been saved to {}.'.format(self.checkpoint_path))

                coord.request_stop()
                coord.join(threads)

        return self.checkpoint_path

    def save(self, model_path):
        if not model_path:
            raise Exception("model path is None.")

        model_dir, model_name = os.path.split(self.checkpoint_path)
        models = [model for model in os.listdir(model_dir) if model_name in model]

        if os.path.splitext(model_path)[-1] != ".zip":
            model_path = os.path.join(model_path, "model.zip")

        if not os.path.isdir(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        with zipfile.ZipFile(model_path, "w") as f:
            for model_file in models:
                model_file_path = os.path.join(model_dir, model_file)
                f.write(model_file_path, model_file, compress_type=zipfile.ZIP_DEFLATED)

        return model_path

    def predict(self, data, input_shape=None, **kwargs):
        if data is None:
            raise Exception("Predict data is None")

        inference_output_dir = os.getenv("RESULT_SAVED_URL")

        with self.tf_graph.as_default():
            if not self.has_fast_rcnn_predict:
                self._fast_rcnn_predict()
                self.has_fast_rcnn_predict = True

            restorer = self._get_restorer()

            config = tf.ConfigProto()
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            with tf.Session(config=config) as sess:
                sess.run(init_op)

                restorer.restore(sess, self.checkpoint_path)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)

                imgs = [cv2.imread(img) for img in data]
                img_names = [os.path.basename(img_path) for img_path in data]

                predict_dict = {}

                for i, img in enumerate(imgs):
                    start = time.time()

                    _img_batch, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category = \
                        sess.run(
                            [self.img_batch, self.fast_rcnn_decode_boxes, self.fast_rcnn_score,
                             self.detection_category],
                            feed_dict={self.img_plac: img})
                    end = time.time()

                    # predict box dict
                    predict_dict[str(img_names[i])] = []

                    for label in NAME_LABEL_MAP.keys():
                        if label == 'back_ground':
                            continue
                        else:
                            temp_dict = {}
                            temp_dict['name'] = label

                            ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
                            temp_boxes = _fast_rcnn_decode_boxes[ind]
                            temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
                            temp_dict['bbox'] = np.array(np.concatenate(
                                [temp_boxes, temp_score], axis=1), np.float64)
                            predict_dict[str(img_names[i])].append(temp_dict)

                    img_np = np.squeeze(_img_batch, axis=0)

                    img_np = draw_box_cv(img_np,
                                         boxes=_fast_rcnn_decode_boxes,
                                         labels=_detection_category,
                                         scores=_fast_rcnn_score)

                    if inference_output_dir:
                        mkdir(inference_output_dir)
                        cv2.imwrite(inference_output_dir + '/{}_fpn.jpg'.format(img_names[i]), img_np)
                        view_bar('{} cost {}s'.format(img_names[i], (end - start)), i + 1, len(imgs))
                        print(f"\nInference results have been saved to directory:{inference_output_dir}.")

                coord.request_stop()
                coord.join(threads)

        return predict_dict

    def load(self, model_url=None):
        if model_url:
            model_dir = os.path.split(model_url)[0]
            with zipfile.ZipFile(model_url, "r") as f:
                f.extractall(path=model_dir)
                ckpt_name = os.path.basename(f.namelist()[0])
                index = ckpt_name.find("ckpt")
                ckpt_name = ckpt_name[:index + 4]
            self.checkpoint_path = os.path.join(model_dir, ckpt_name)

        else:
            raise Exception(f"model url is None")

        return self.checkpoint_path

    def evaluate(self, data, model_path, **kwargs):
        if data is None or data.x is None or data.y is None:
            raise Exception("Prediction data is None")

        self.load(model_path)
        predict_dict = self.predict(data.x)
        metric_name, metric_func = kwargs.get("metric")
        if callable(metric_func):
            return {"f1_score": metric_func(data.y, predict_dict)}
        else:
            raise Exception(f"not found model metric func(name={metric_name}) in model eval phase")

    def _get_restorer(self):
        model_variables = slim.get_model_variables()
        restore_variables = [var for var in model_variables if not var.name.startswith(
            'Fast_Rcnn')] + [tf.train.get_or_create_global_step()]
        return tf.train.Saver(restore_variables)

    def _init_tf_graph(self):
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

            self.img_tensor = tf.cast(self.img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
            self.img_batch = image_preprocess.short_side_resize_for_inference_data(self.img_tensor,
                                                                                   target_shortside_len=cfgs.SHORT_SIDE_LEN,
                                                                                   is_resize=True)

    def _fast_rcnn_predict(self):
        with self.tf_graph.as_default():
            # ***********************************************************************************************
            # *                                         share net                                           *
            # ***********************************************************************************************
            _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                              inputs=self.img_batch,
                                              num_classes=None,
                                              is_training=True,
                                              output_stride=None,
                                              global_pool=False,
                                              spatial_squeeze=False)
            # ***********************************************************************************************
            # *                                            RPN                                              *
            # ***********************************************************************************************
            rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                                inputs=self.img_batch,
                                gtboxes_and_label=None,
                                is_training=False,
                                share_head=cfgs.SHARE_HEAD,
                                share_net=share_net,
                                stride=cfgs.STRIDE,
                                anchor_ratios=cfgs.ANCHOR_RATIOS,
                                anchor_scales=cfgs.ANCHOR_SCALES,
                                scale_factors=cfgs.SCALE_FACTORS,
                                base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                                level=cfgs.LEVEL,
                                top_k_nms=cfgs.RPN_TOP_K_NMS,
                                rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                                max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                                rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                                rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                                rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                                rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                                remove_outside_anchors=False,  # whether remove anchors outside
                                rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

            # rpn predict proposals
            rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

            # ***********************************************************************************************
            # *                                         Fast RCNN                                           *
            # ***********************************************************************************************
            fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=self.img_batch,
                                                 feature_pyramid=rpn.feature_pyramid,
                                                 rpn_proposals_boxes=rpn_proposals_boxes,
                                                 rpn_proposals_scores=rpn_proposals_scores,
                                                 img_shape=tf.shape(self.img_batch),
                                                 roi_size=cfgs.ROI_SIZE,
                                                 scale_factors=cfgs.SCALE_FACTORS,
                                                 roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                                 gtboxes_and_label=None,
                                                 fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                                 fast_rcnn_maximum_boxes_per_img=100,
                                                 fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                                 show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                                 # show detections which score >= 0.6
                                                 num_classes=cfgs.CLASS_NUM,
                                                 fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                                 fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                                 fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                                 use_dropout=False,
                                                 weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                                 is_training=False,
                                                 level=cfgs.LEVEL)

            self.fast_rcnn_decode_boxes, self.fast_rcnn_score, self.num_of_objects, self.detection_category = \
                fast_rcnn.fast_rcnn_predict()
