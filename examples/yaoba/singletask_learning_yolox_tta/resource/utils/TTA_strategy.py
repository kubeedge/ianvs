import os.path
import warnings

warnings.filterwarnings("ignore")
from .general_TTA_v5 import *
from itertools import zip_longest, combinations, permutations
import sys

sys.path.append("/home/wjj/wjj/Public/code/huawei")

augment_list = TTA_Aug_List()


class TTA_Strategy(object):
    def __init__(self, model, val_image_path, val_anno_path, log_dir, worker=4, nms_thr=0.5):
        self.default_img_size = None  # will override by _prepare_for_search()
        self.model = self._prepare_for_search(model, val_anno_path, val_image_path)
        self.val_image_path = val_image_path
        self.val_anno_path = val_anno_path
        self.augments = TTA_Aug_List()
        self.augments_space = TTA_Aug_Space(resolution=self.default_img_size, size_divisor=32)
        # self.augments_space =Test_Aug_Space()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_dir = log_dir
        self.nms_thr = nms_thr
        self.worker = worker

    def search_single_strategy(self, top_num):
        val_aug_results = []
        log_txt = open(os.path.join(self.log_dir, "single_log.txt"), 'w')
        for idx, parameter in tqdm(self.augments_space):
            method = self.augments[idx]
            v = parameter
            if method.__name__ != 'TTA_Resize':
                augs = [[('TTA_Resize', self.default_img_size)],
                        [('TTA_Resize', self.default_img_size), (method.__name__, v)], ]
            else:
                augs = [[('TTA_Resize', self.default_img_size)],
                        [(method.__name__, v)], ]
            print(augs)
            ap, _ = model_TTA_infer(model=self.model,
                                    img_path=self.val_image_path,
                                    anno_path=self.val_anno_path,
                                    augs=augs,
                                    worker=self.worker,
                                    nms_thr=self.nms_thr)
            log_txt.write(str(augs) + ' ' + str(round(ap, 3)) + '\n')
            val_aug_results.append(ap)
        val_aug_results_np = np.array(val_aug_results)
        sort_idx = np.flipud(val_aug_results_np.argsort())
        single_method_idx_top = []
        single_method_name_top = []
        single_factor_top = []
        single_ap_top = []
        for idx in sort_idx:
            method_index = self.augments_space[idx][0]
            method_name = self.augments[self.augments_space[idx][0]].__name__
            v = self.augments_space[idx][1]
            ap = val_aug_results_np[idx]
            if method_index not in single_method_idx_top:
                single_method_idx_top.append(method_index)
                single_method_name_top.append(method_name)
                single_factor_top.append(v)
                single_ap_top.append(ap)
            else:
                site = single_method_idx_top.index(method_index)
                if ap > single_ap_top[site]:
                    single_factor_top[site] = v
                    single_ap_top[site] = ap
            if len(single_method_idx_top) == top_num:
                break
        self._write_single_log(single_method_name_top, single_factor_top, single_ap_top, "search_single_strategy.txt")
        return list(zip_longest(single_method_name_top, single_factor_top))

    def search_cascade_strategy(self, single_top_strategies, cascade_num=3, top_num=5):
        cascade_strategies = []
        for num in range(1, cascade_num + 1):
            cascade_strategies.extend(list(permutations(single_top_strategies, num)))
        cascade_ap = []
        strategy_log = []
        for strategy in tqdm(cascade_strategies):
            strategy_list = list(strategy)
            tmp = False
            for s in strategy_list:
                if s[0] == 'TTA_Resize':
                    tmp = True
            augs = [[('TTA_Resize', self.default_img_size)]]
            if tmp:
                augs.append(strategy_list)
            else:
                strategy_list.insert(0, ('TTA_Resize', self.default_img_size))
                augs.append(strategy_list)
            strategy_log.append(strategy_list)
            print(augs)
            ap, _ = model_TTA_infer(model=self.model,
                                    img_path=self.val_image_path,
                                    anno_path=self.val_anno_path,
                                    augs=augs,
                                    worker=self.worker,
                                    nms_thr=self.nms_thr
                                    )
            cascade_ap.append(ap)
        cascade_ap_np = np.array(cascade_ap)
        sort_idx = np.flipud(cascade_ap_np.argsort())
        cascade_method_top = []
        cascade_ap_top = []
        for idx in sort_idx[:top_num]:
            cascade_method_top.append(strategy_log[idx])
            cascade_ap_top.append(cascade_ap[idx])
        self._write_cascade_log(cascade_method_top, cascade_ap_top, "search_cascade_strategy.txt")
        return cascade_method_top

    def search_combine_single_strategy(self, single_top_strategies, top_num, ):
        combine_strategies = []
        for num in range(2, len(single_top_strategies) + 1):
            for combine_strategy in combinations(single_top_strategies, num):
                combine_strategies.append(list(combine_strategy))
        log_txt = open(os.path.join(self.log_dir, "combine_single_log.txt"), 'w')
        combine_single_ap = []
        combine_strategies_log = []
        for c_s in tqdm(combine_strategies):
            augs = [[('TTA_Resize', self.default_img_size)]]
            for s in c_s:
                if s[0] == 'TTA_Resize':
                    augs.append([s])
                else:
                    new_s = [('TTA_Resize', self.default_img_size)]
                    new_s.append(s)
                    augs.append(new_s)
            print(augs)
            combine_strategies_log.append(augs)
            ap, _ = model_TTA_infer(model=self.model,
                                    img_path=self.val_image_path,
                                    anno_path=self.val_anno_path,
                                    augs=augs,
                                    worker=self.worker,
                                    nms_thr=self.nms_thr
                                    )
            combine_single_ap.append(ap)
            log_txt.write(str(augs) + ' ' + '\n')
        combine_single_ap_np = np.array(combine_single_ap)
        sort_idx = np.flipud(combine_single_ap_np.argsort())
        combine_single_method_top = []
        combine_single_ap_top = []
        for idx in sort_idx[:top_num]:
            combine_single_method_top.append(combine_strategies_log[idx])
            combine_single_ap_top.append(combine_single_ap[idx])
        self._write_combine_log(combine_single_method_top, "search_combine_single_strategy.txt")
        return combine_single_method_top

    def _prepare_for_search(self, model, val_anno_path, val_img_prefix):
        model_cfg = copy.deepcopy(model.cfg)

        # Get the default image resolution of the model
        for p in model_cfg.data.test.pipeline:
            if p['type'] == 'MultiScaleFlipAug':
                self.default_img_size = p['img_scale']
        if not hasattr(self, "default_img_size"):
            raise ValueError("can not find img_scale model cfg")

        # Override the data.test part in cfg with the validation set info.
        model_cfg.data.test.ann_file = val_anno_path
        model_cfg.data.test.img_prefix = val_img_prefix
        test_transforms = model_cfg.data.test.pipeline[1].transforms

        # Remove the Resize and flip operations from the original test_pipeline.
        new_test_transforms = [i for i in test_transforms if i['type'] not in ['Resize', 'RandomFlip']]
        model_cfg.data.test.pipeline[1].transforms = new_test_transforms
        model.cfg = model_cfg
        return model

    def _write_single_log(self, method_name_list, method_factor_list, method_ap_list, out_name):
        fp = open(os.path.join(self.log_dir, out_name), 'w')
        for i in range(len(method_name_list)):
            fp.write(f"{method_name_list[i]} {method_factor_list[i]} {method_ap_list[i]}\n")
        fp.close()

    def _write_cascade_log(self, method_name_list, method_ap_list, out_name):
        fp = open(os.path.join(self.log_dir, out_name), 'w')
        for i in range(len(method_name_list)):
            fp.write(f"{method_name_list[i]} {method_ap_list[i]}\n")
        fp.close()

    def _write_combine_log(self, method_idx_list, out_name):
        fp = open(os.path.join(self.log_dir, out_name), 'w')
        for method_sequence in method_idx_list:
            for method in method_sequence:
                name, factor = method[0], method[1]
                method_idx = -1
                for i in range(len(augment_list)):
                    if augment_list[i].__name__ == name:
                        method_idx = i
                if isinstance(factor, tuple):
                    factor = f"{factor[0]}.{factor[1]}"
                else:
                    factor = str(factor)
                fp.write(f"{str(method_idx)},{factor};")
            fp.write("\n")
