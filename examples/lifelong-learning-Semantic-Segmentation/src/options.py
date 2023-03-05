from __future__ import absolute_import, division, print_function

import argparse


class ERFnetOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)
        self.parser.add_argument("--cluster_mode",
                                 type=str,
                                 help="name of the cluster",
                                 choices=['laptop', 'cluster', 'phoenix'],
                                 default=None)

        # DATA options
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to use",
                                 choices=['KITTI', 'KITTI_2015', 'cityscapes', 'mapillary', 'mapillary_by_ID'],
                                 default='cityscapes')
        self.parser.add_argument("--dataset_split",
                                 type=str,
                                 help="split of the dataset to use, can be none if there is no special split",
                                 #choices=['res_288x960', 'res_512x1024_stage2', 'res_512x1024_stage3', 'res_512x1024_stage4'],
                                 default=None)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=512)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=1024)
        self.parser.add_argument("--crop_height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--crop_width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
        self.parser.add_argument("--video_frames",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0])
        self.parser.add_argument("--single_set",
                                 help="do not use an incremental dataset ClassDefinition",
                                 action="store_true")
        self.parser.add_argument("--hyperparameter",
                                 help="load hyperparameter search sets",
                                 action="store_true")

        # MODEL options
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--sigmoid",
                                 help="use logistic sigmoid instead of softmax",
                                 action="store_true")

        # LOADING options
        self.parser.add_argument("--load_model_name",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--weights_epoch",
                                 type=str,
                                 help="name of model to load")

        # TRAINING options
        self.parser.add_argument("--train_set",
                                 help="select train set",
                                 type=int,
                                 default=123)
        self.parser.add_argument("--city",
                                 help="apply city filter",
                                 action="store_true")
        self.parser.add_argument("--temp",
                                 help="distillation temperature",
                                 type=int,
                                 default=1)
        self.parser.add_argument("--n_files",
                                 help="number of files to load by dataloader",
                                 type=int)
        self.parser.add_argument("--teachers",
                                 help="teachers to use - order: name1 epoch1 trainset1 name2 epoch2 trainset2 ...",
                                 nargs='+')
        self.parser.add_argument("--lambda_GS",
                                 help="lambda GS of the distillation loss",
                                 type=int,
                                 default=1)
        
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=5e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=50)
        self.parser.add_argument("--weight_decay",
                                 help="weight decay",
                                 type=float,
                                 default=3e-4)

        # LOGGING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="test_erfnet")
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=10)

        # EVALUATION options
        self.parser.add_argument("--save_pred_segs",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--save_pred_to_disk",
                                 help="save the predictions to disk",
                                 action="store_true")
        self.parser.add_argument("--pred_frequency",
                                 help="number of images between each exported segmentation image",
                                 type=int,
                                 default=25)
        self.parser.add_argument("--pred_wout_blend",
                                 help="do not blend network output with void map",
                                 action="store_true")
        self.parser.add_argument("--validate",
                                 help="validate model after each epoch",
                                 action="store_true")
        self.parser.add_argument("--val_frequency",
                                 type=int,
                                 help="number of epochs between each validation. For standalone, any number > 0 will produce an output",
                                 default=1)
        self.parser.add_argument("--save_probs_to_disk",
                                 help="save the class probabilities of every class-feature-map to disk",
                                 action="store_true")
        self.parser.add_argument("--save_entropy_to_disk",
                                 help="save the entropy maps to disk, uses --probs_frequency",
                                 action="store_true")
        self.parser.add_argument("--probs_frequency",
                                 help="number of images between each exported probs",
                                 type=int,
                                 default=25)
        self.parser.add_argument("--task_to_val",
                                 help="on which train set (task) should be validated",
                                 type=int,
                                 default=0)
        self.parser.add_argument("--mean_entropy",
                                 help="calculate the mean entropy per class",
                                 action="store_true")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


    def parse_str(self, str):
        self.options = self.parser.parse_args(str)
        return self.options
