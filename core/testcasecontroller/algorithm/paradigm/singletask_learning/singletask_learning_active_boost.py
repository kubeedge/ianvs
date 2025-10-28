# pylint: disable=C0301
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=W0246
# pylint: disable=R1725
# pylint: disable=R1732
# pylint: disable=R0913
# pylint: disable=R0801
import json
import os
import os.path as osp
from core.common.constant import ParadigmType
from examples.yaoba.singletask_learning_boost.resource.utils.infer_and_error import infer_anno, merge_predict_results, \
    compute_error, gen_txt_according_json, get_new_train_json
from examples.yaoba.singletask_learning_boost.resource.utils.transform_unkonwn import aug_image_bboxes
from .singletask_learning import SingleTaskLearning


class SingleTaskLearningACBoost(SingleTaskLearning):

    def __init__(self, workspace, **kwargs):
        super(SingleTaskLearningACBoost, self).__init__(workspace, **kwargs)

    def run(self):
        job = self.build_paradigm_job(str(ParadigmType.SINGLE_TASK_LEARNING.value))
        known_dataset_json, unknown_dataset_json, img_path = self._prepare_for_calculate_weights()
        base_config_path = osp.join(job.resource_dir, "base_config.py")
        train_script_path = osp.join(job.resource_dir, "train.py")
        ac_boost_training_json, aug_img_folder = self._calculate_weights_for_training(
            base_config=base_config_path,
            known_json_path=known_dataset_json,
            unknown_json_path=unknown_dataset_json,
            img_path=img_path,
            tmp_path=os.path.join(job.work_dir, "tmp_folder"),
            train_script_path=train_script_path
        )
        trained_model = self._ac_boost_train(job, ac_boost_training_json, aug_img_folder)
        inference_result = self._inference(job, trained_model)
        self.system_metric_info['use_raw'] = True
        return inference_result, self.system_metric_info

    def _ac_boost_train(self, job, training_anno, training_img_folder):
        train_output_model_path = job.train((training_img_folder, training_anno))
        trained_model_path = job.save(train_output_model_path)
        return trained_model_path

    def _inference(self, job, trained_model):
        # Load test set data
        img_prefix = self.dataset.image_folder_url
        ann_file_path = self.dataset.test_url
        ann_file = json.load(open(ann_file_path, mode="r", encoding="utf-8"))
        test_set = []
        for i in ann_file['images']:
            test_set.append(os.path.join(img_prefix, i['file_name']))

        job.load(trained_model)
        infer_res = job.predict(test_set)
        return infer_res

    def _prepare_for_calculate_weights(self):
        known_dataset_json = self.dataset.known_dataset_url
        unknown_dataset_json = self.dataset.unknown_dataset_url
        img_path = self.dataset.image_folder_url
        return known_dataset_json, unknown_dataset_json, img_path

    def _calculate_weights_for_training(self,
                                        base_config,
                                        known_json_path,
                                        unknown_json_path,
                                        img_path,
                                        tmp_path,
                                        train_script_path):
        r"""Generate instance weights required for unknown task training. In object detection,
            an instance means a bounding box, i.e., generating training weights for each bounding box.
        Args:
            base_config (str): path of config file for training known/unknown model
            known_json_path (str): path of JSON file for training known model
            unknown_json_path (str): path of JSON file for training unknown model
            img_path (str): image path of training, validation, and test set.
            tmp_path (str): path to save temporary files, including augmented images, training JSON files, etc.
            train_script_path (str): path of mmdet training script
        Return:
            new_training_weight (str): JSON file with instance weights for unknown task training,
                which contains both the known and unknown training sets.
            aug_img_folder (str): the image paths required for training the model using the JSON file with instance weights.
        """
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        # Define necessary path
        aug_img_folder = osp.join(tmp_path, "aug_img_folder")  # The directory for saving augmented images
        known_model_folder = osp.join(tmp_path, "known_model")  # The directory for saving known model training results
        unknown_model_folder = osp.join(tmp_path, "unknown_model")  # The directory for saving unknown model training results
        aug_unknown_json = osp.join(tmp_path, 'aug_unknown.json')  # The JSON file path of the unknown data after augmentation

        # Augmenting the unknown data and returning the paths of the augmented images
        aug_image_bboxes(
            anno=unknown_json_path,
            augs=[('flip', 1), ('brightness', 0.6), ('flip', -1)],
            image_path=img_path,
            out_path=tmp_path
        )

        # Train the known model
        known_model_training_task = f"python {train_script_path} " \
                                    f"{base_config} --seed 1 --deterministic --cfg-options " \
                                    f"data.train.ann_file={known_json_path} " \
                                    f"data.train.img_prefix={img_path} " \
                                    f"work_dir={known_model_folder}"
        os.system(known_model_training_task)

        # Train the unknown model
        unknown_model_training_task = f"python {train_script_path} " \
                                      f"{base_config} --seed 1 --deterministic --cfg-options " \
                                      f"data.train.ann_file={aug_unknown_json} " \
                                      f"data.train.img_prefix={aug_img_folder} " \
                                      f"work_dir={unknown_model_folder}"
        os.system(unknown_model_training_task)

        # using above known model to infer unknown data
        infer_anno(
            config_file=base_config,
            checkpoint_file=osp.join(known_model_folder, 'latest.pth'),
            img_path=aug_img_folder,
            anno_path=aug_unknown_json,
            out_path=osp.join(tmp_path, 'unknown_infer_results.json')
        )

        # using above unknown model to infer known data
        infer_anno(
            config_file=base_config,
            checkpoint_file=osp.join(unknown_model_folder, 'latest.pth'),
            img_path=aug_img_folder,
            anno_path=known_json_path,
            out_path=osp.join(tmp_path, 'known_infer_results.json')
        )

        # merging the prediction results and computing error
        merge_predict_results(
            result1=osp.join(tmp_path, 'unknown_infer_results.json'),
            result2=osp.join(tmp_path, 'known_infer_results.json'),
            out_dir=osp.join(tmp_path, "merge_predict_result.json")
        )
        new_json = compute_error(osp.join(tmp_path, "merge_predict_result.json"))

        # generating the weights of the overall training sample based on the  prediction error.
        gen_txt_according_json(known_json_path, osp.join(tmp_path, 'known.txt'))
        gen_txt_according_json(aug_unknown_json, osp.join(tmp_path, 'aug_unknown.txt'))
        get_new_train_json(
            new_json,
            aug_img_folder,
            osp.join(tmp_path, 'known.txt'),
            osp.join(tmp_path, 'aug_unknown.txt'),
            out_dir=osp.join(tmp_path, 'new_training_weight.json'))

        return osp.join(tmp_path, 'new_training_weight.json'), aug_img_folder
