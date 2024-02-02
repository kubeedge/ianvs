import torch
import sys

sys.path.append('/home/wjj/wjj/Public/code/ianvs')
from mmdet.apis import init_detector

from custom_ianvs.test_time_aug.TTA_augs_xyxy_cv2 import TTA_Aug_List, TTA_Aug_Space
from custom_ianvs.test_time_aug.TTA_strategy import TTA_Strategy
from custom_ianvs.test_time_aug.general_TTA_v5 import model_TTA_infer

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    yolo_model = init_detector(r"/home/wjj/wjj/Public/code/ianvs/custom_ianvs/mmdet_related/yolox_s_8x8_300e_yaoba.py",
                               r"/home/wjj/wjj/Public/code/ianvs/custom_ianvs/mmdet_related/epoch_300.pth",
                               'cuda:0')
    search_agent = TTA_Strategy(model=yolo_model,
                                val_image_path="/media/huawei_YaoBa/Images",
                                val_anno_path="/home/wjj/wjj/Public/code/ianvs/dataset/yaoba/yaoba_tta/val.json",
                                augments=TTA_Aug_List(),
                                augments_space=TTA_Aug_Space(),
                                log_dir="/home/wjj/wjj/Public/code/ianvs/custom_ianvs/test_time_aug/log",
                                worker=6
                                )
    search_agent.search_single_strategy(top_num=5)
    single_strategies = [('TTA_Resize', (800, 800)),
                         ('TTA_Flip', -1),
                         ('TTA_AutoContrast', 0.3),
                         ('TTA_Sharpness', 0.5),
                         ('TTA_Brightness', 0.7)]
    search_agent.search_cascade_strategy(single_strategies, cascade_num=3, top_num=5)
    search_agent.search_combine_single_strategy(single_strategies, top_num=3)
    # 多进程推理
    model_TTA_infer(model=yolo_model,
                    img_path="/media/huawei_YaoBa/Images",
                    anno_path="/home/wjj/wjj/Public/code/ianvs/dataset/yaoba/yaoba_tta/test.json",
                    augs=[
                        [('TTA_Resize', (640, 640))],
                        [('TTA_Brightness', (640, 640)), ]
                    ],
                    worker=4,
                    nms_thr=0.5)
