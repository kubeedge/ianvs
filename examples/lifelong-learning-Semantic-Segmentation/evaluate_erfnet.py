from __future__ import absolute_import, division, print_function
import os
import cv2
import random
import json
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import h5py as h5

from dataloader.pt_data_loader import mytransforms
from dataloader.pt_data_loader.specialdatasets import CityscapesDataset
from models.erfnet import ERFNet
from dataloader.eval.metrics import SegmentationRunningScore
from dataloader.file_io.get_path import GetPath
from dataloader.definitions.labels_file import *
from src.options import ERFnetOptions
from src.city_set import CitySet

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
os.environ['PYTHONHASHSEED'] = '0'
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # Romera
torch.cuda.manual_seed_all(seed)  # Romera
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Evaluator:
    def __init__(self, options, model=None):

        if __name__ == "__main__":
            print(" -> Executing script", os.path.basename(__file__))

        self.opt = options
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           LABELS
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert self.opt.train_set in {1, 2, 3, 12, 123}, "Invalid train_set!"
        assert self.opt.task_to_val in {0, 1, 2, 3, 12, 123}, "Invalid task!"
        keys_to_load = ['color', 'segmentation']

        # Labels
        labels = self._get_labels_cityscapes()

        # Train IDs
        self.train_ids = set([labels[i].trainId for i in range(len(labels))])
        self.train_ids.remove(255)
        self.train_ids = sorted(list(self.train_ids))

        self.num_classes_model = len(self.train_ids)

        # Task handling
        if self.opt.task_to_val != 0:
            labels_task = self._get_task_labels_cityscapes()
            train_ids_task = set([labels_task[i].trainId for i in range(len(labels_task))])
            train_ids_task.remove(255)
            self.task_low = min(train_ids_task)
            self.task_high = max(train_ids_task) + 1
            labels = labels_task
            self.train_ids = sorted(list(train_ids_task))
        else:
            self.task_low = 0
            self.task_high = self.num_classes_model
            self.opt.task_to_val = self.opt.train_set

        # Number of classes for the SegmentationRunningScore
        self.num_classes_score = self.task_high - self.task_low


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           DATASET DEFINITIONS
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Data augmentation
        test_data_transforms = [mytransforms.CreateScaledImage(),
                                mytransforms.Resize((self.opt.height, self.opt.width), image_types=['color']),
                                mytransforms.ConvertSegmentation(),
                                mytransforms.CreateColoraug(new_element=True, scales=self.opt.scales),
                                mytransforms.RemoveOriginals(),
                                mytransforms.ToTensor(),
                                mytransforms.NormalizeZeroMean(),
                                ]

        # If hyperparameter search, only load the respective validation set. Else, load the full validation set.
        if self.opt.hyperparameter:
            trainvaltest_split = 'train'
            folders_to_load = CitySet.get_city_set(-1)
        else:
            trainvaltest_split = 'validation'
            folders_to_load = None

        test_dataset = CityscapesDataset(dataset='cityscapes',
                                         split=self.opt.dataset_split,
                                         trainvaltest_split=trainvaltest_split,
                                         video_mode='mono',
                                         stereo_mode='mono',
                                         scales=self.opt.scales,
                                         labels_mode='fromid',
                                         labels=labels,
                                         keys_to_load=keys_to_load,
                                         data_transforms=test_data_transforms,
                                         video_frames=self.opt.video_frames,
                                         folders_to_load=folders_to_load)

        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.opt.batch_size,
                                      shuffle=False,
                                      num_workers=self.opt.num_workers,
                                      pin_memory=True,
                                      drop_last=False)

        print("++++++++++++++++++++++ INIT VALIDATION ++++++++++++++++++++++++")
        print("Using dataset\n  ", self.opt.dataset, "with split", self.opt.dataset_split)
        print("There are {:d} validation items\n  ".format(len(test_dataset)))
        print("Validating classes up to train set\n  ", self.opt.train_set)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           LOGGING OPTIONS
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If no model is passed, standalone validation is to be carried out. The log_path needs to be set before
        # self.load_model() is invoked.
        if model is None:
            self.opt.validate = False
            self.opt.model_name = self.opt.load_model_name

        path_getter = GetPath()
        log_path = path_getter.get_checkpoint_path()
        self.log_path = os.path.join(log_path, 'erfnet', self.opt.model_name)

        # All outputs will be saved to save_path
        self.save_path = self.log_path

        # Create output path for standalone validation
        if not self.opt.validate:
            save_dir = 'eval_{}'.format(self.opt.dataset)

            if self.opt.hyperparameter:
                save_dir = save_dir + '_hyper'

            save_dir = save_dir + '_task_to_val{}'.format(self.opt.task_to_val)

            self.save_path = os.path.join(self.log_path, save_dir)

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # Copy this file to save_path
        shutil.copy2(__file__, self.save_path)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           MODEL DEFINITION
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Standalone validation
        if not self.opt.validate:
            # Create a conventional ERFNet
            self.model = ERFNet(self.num_classes_model, self.opt)
            self.load_model()
            self.model.to(self.device)

        # Validate while training
        else:
            self.model = model

        self.model.eval()

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #                           LOGGING OPTIONS II
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # self.called is used to decide which file mode shall be used when writing metrics to disk.
        self.called = False

        self.metric_model = SegmentationRunningScore(self.num_classes_score)

        # Metrics are only saved if val_frequency > 0!
        if self.opt.val_frequency != 0:
            print("Saving metrics to\n  ", self.save_path)

        # Set up colour output. Coloured images are only output if standalone validation is carried out!
        if not self.opt.validate and self.opt.save_pred_to_disk:
            # Output path
            self.img_path = os.path.join(self.save_path, 'output_{}'.format(self.opt.weights_epoch))

            if self.opt.pred_wout_blend:
                self.img_path += '_wout_blend'

            if not os.path.exists(self.img_path):
                os.makedirs(self.img_path)
            print("Saving prediction images to\n  ", self.img_path)
            print("Save frequency\n  ", self.opt.pred_frequency)

            # Get the colours from dataset.
            colors = [(label.trainId - self.task_low, label.color) for label in labels if
                      label.trainId != 255 and label.trainId in self.train_ids]
            colors.append((255, (0, 0, 0)))  # void class
            self.id_color = dict(colors)
            self.id_color_keys = [key for key in self.id_color.keys()]
            self.id_color_vals = [val for val in self.id_color.values()]

            # Ongoing index to name the outputs
            self.img_idx = 0

        # Set up probability output. Probabilities are only output if standalone validation is carried out!
        if not self.opt.validate and self.opt.save_probs_to_disk:
            # Output path
            self.logit_path = os.path.join(self.save_path, 'probabilities_{}'.format(self.opt.weights_epoch))
            if not os.path.exists(self.logit_path):
                os.makedirs(self.logit_path)
            print("Saving probabilities to\n  ", self.logit_path)
            print("Save frequency\n  ", self.opt.probs_frequency)

            # Ongoing index to name the probability outputs
            self.probs_idx = 0

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # Save all options to disk and print them to stdout
        self._print_options()
        self._save_opts(len(test_dataset))

    def _get_labels_cityscapes(self, id=None):
        if id is None:
            id = self.opt.train_set

        if id == 1:
            labels = labels_cityscape_seg_train1.getlabels()
        elif id == 2:
            labels = labels_cityscape_seg_train2_eval.getlabels()
        elif id == 12:
            labels = labels_cityscape_seg_train2_eval.getlabels()
        elif id == 3:
            labels = labels_cityscape_seg_train3_eval.getlabels()
        elif id == 123:
            labels = labels_cityscape_seg_train3_eval.getlabels()

        return labels

    def _get_task_labels_cityscapes(self, id=None):
        if id is None:
            id = self.opt.task_to_val

        if id == 1:
            labels_task = labels_cityscape_seg_train1.getlabels()
        elif id == 2:
            labels_task = labels_cityscape_seg_train2.getlabels()
        elif id == 12:
            labels_task = labels_cityscape_seg_train2_eval.getlabels()
        elif id == 3:
            labels_task = labels_cityscape_seg_train3.getlabels()
        elif id == 123:
            labels_task = labels_cityscape_seg_train3_eval.getlabels()

        return labels_task

    def load_model(self):
        """Load model(s) from disk
        """
        base_path = os.path.split(self.log_path)[0]
        checkpoint_path = os.path.join(base_path, self.opt.load_model_name, 'models',
                                       'weights_{}'.format(self.opt.weights_epoch))
        assert os.path.isdir(checkpoint_path), \
            "Cannot find folder {}".format(checkpoint_path)
        print("loading model from folder {}".format(checkpoint_path))

        path = os.path.join(checkpoint_path, "{}.pth".format('model'))
        model_dict = self.model.state_dict()
        if self.opt.no_cuda:
            pretrained_dict = torch.load(path, map_location='cpu')
        else:
            pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def calculate_metrics(self, epoch=None):
        print("-> Computing predictions with input size {}x{}".format(self.opt.height, self.opt.width))
        print("-> Evaluating")

        for data in self.test_loader:
            with torch.no_grad():
                input_color = data[("color_aug", 0, 0)]
                gt_seg = data[('segmentation', 0, 0)][:, 0, :, :].numpy()
                input_color = {("color_aug", 0, 0): input_color.to(self.device)}

                output = self.model(input_color)

                pred_seg = output['segmentation_logits'].float()

                # Apply task reduction for argmax
                if self.opt.task_to_val != 0:
                    pred_seg = pred_seg[:, self.task_low:self.task_high, ...]
                    gt_seg -= self.task_low  # gt_seg trainIDs must be in range(0, self.num_classes_score) to map them with torch.argmax output
                    gt_seg[gt_seg == 255 - self.task_low] = 255  # maintaining the background trainID

                # Save probabilities to disk
                if not self.opt.validate and self.opt.save_probs_to_disk:
                    self._save_probs_to_disk(F.softmax(pred_seg, dim=1).cpu().numpy())

                pred_seg = F.interpolate(pred_seg, gt_seg[0].shape, mode='nearest')

                # Select most probable class
                pred_seg = torch.argmax(pred_seg, dim=1)

                pred_seg = pred_seg.cpu().numpy()
                self.metric_model.update(gt_seg, pred_seg)

                # Save predictions to disk
                if not self.opt.validate and self.opt.save_pred_to_disk:
                    self._save_pred_to_disk(pred_seg, gt_seg)

        metrics = self.metric_model.get_scores()

        # Save metrics
        if self.opt.val_frequency != 0:
            # Local epoch will not be specified if the validation is carried out standalone.
            if not self.opt.validate and epoch is None:
                epoch = int(self.opt.weights_epoch)

            self._save_metrics(epoch, metrics)

        self.metric_model.reset()
        print("\n  " + ("{:>8} | " * 2).format("miou", "maccuracy"))
        print(("&{: 8.3f}  " * 2).format(metrics['meaniou'], metrics['meanacc']) + "\\\\")
        print("\n-> Done!")


    def _save_metrics(self, epoch, metrics):
        ''' Save metrics (class-wise) to disk as HDF5 file.
        '''
        # If a single model is validated, the output file will carry its epoch number in its file name. If a learning
        # process is validated "on the go", the output filename will just be "validation.h5".
        if not self.opt.validate:
            filename = 'validation_{:d}.h5'.format(epoch)
        else:
            filename = 'validation.h5'
        save_path = os.path.join(self.save_path, filename)

        # When _save_metrics is invoked for the first time, the HDF file will be opened in "w" mode overwriting any
        # existing file. In case of another invocation, the file will be opened in "a" mode not overwriting any
        # existing file but appending the data.
        if not self.called:
            mode = 'w'
            self.called = True
        else:
            mode = 'a'

        # If a single model is validated, all datasets reside in the first layer of the HDF file. If a learning process
        # is validated "on the go", each validated model will have its own group named after the epoch of the model.
        with h5.File(save_path, mode) as f:
            if self.opt.validate:
                grp = f.create_group('epoch_{:d}'.format(epoch))
            else:
                grp = f

            # Write mean_IoU, mean_acc and mean prec to file / group
            dset = grp.create_dataset('mean_IoU', data=metrics['meaniou'])
            dset.attrs['Description'] = 'See trainIDs for information on the classes'
            dset = grp.create_dataset('mean_recall', data=metrics['meanacc'])
            dset.attrs['Description'] = 'See trainIDs for information on the classes'
            dset.attrs['AKA'] = 'Accuracy -> TP / (TP + FN)'
            dset = grp.create_dataset('mean_precision', data=metrics['meanprec'])
            dset.attrs['Description'] = 'See trainIDs for information on the classes'
            dset.attrs['AKA'] = 'Precision -> TP / (TP + FP)'

            # If in 'w' mode, allocate memory for class_id dataset
            if mode == 'w':
                ids = np.zeros(shape=(len(metrics['iou'])), dtype=np.uint32)

            class_iou = np.zeros(shape=(len(metrics['iou'])), dtype=np.float64)
            class_acc = np.zeros(shape=(len(metrics['acc'])), dtype=np.float64)
            class_prec = np.zeros(shape=(len(metrics['prec'])), dtype=np.float64)

            # Disassemble the dictionary
            for key, i in zip(sorted(metrics['iou']), range(len(metrics['iou']))):
                if mode == 'w':
                    ids[i] = self.train_ids[i]  # int(key)
                class_iou[i] = metrics['iou'][key]
                class_acc[i] = metrics['acc'][key]
                class_prec[i] = metrics['prec'][key]

            # Create class_id dataset only once in first layer of HDF5 file when in 'w' mode
            if mode == 'w':
                dset = f.create_dataset('trainIDs', data=ids)
                dset.attrs['Description'] = 'trainIDs of classes'
                dset = f.create_dataset('first_epoch_in_file', data=np.array([epoch]).astype(np.uint32))
                dset.attrs['Description'] = 'First epoch that has been saved in this file.'

            dset = grp.create_dataset('class_IoU', data=class_iou)
            dset.attrs['Description'] = 'See trainIDs for information on the class order'
            dset = grp.create_dataset('class_recall', data=class_acc)
            dset.attrs['Description'] = 'See trainIDs for information on the class order'
            dset.attrs['AKA'] = 'Accuracy -> TP / (TP + FN)'
            dset = grp.create_dataset('class_precision', data=class_prec)
            dset.attrs['Description'] = 'See trainIDs for information on the class order'
            dset.attrs['AKA'] = 'Precision -> TP / (TP + FP)'


    def _save_pred_to_disk(self, pred, gt):
        ''' Save a correctly coloured image of the prediction (batch) to disk. Only every self.opt.pred_frequency-th
            prediction is saved to disk!
        '''
        for i in range(gt.shape[0]):
            if self.img_idx % self.opt.pred_frequency == 0:
                o_size = gt[i].shape  # original image shape

                single_pred = pred[i].flatten()
                single_gt = gt[i].flatten()

                # Copy voids from ground truth to prediction
                if not self.opt.pred_wout_blend:
                    single_pred[single_gt == 255] = 255

                # Convert to colour
                single_pred = self._convert_to_colour(single_pred, o_size)
                single_gt = self._convert_to_colour(single_gt, o_size)

                # Save predictions to disk using an ongoing index
                cv2.imwrite(os.path.join(self.img_path, 'pred_val_{}.png'.format(self.img_idx)), single_pred)
                cv2.imwrite(os.path.join(self.img_path, 'gt_val_{}.png'.format(self.img_idx)), single_gt)

            self.img_idx += 1

    def _convert_to_colour(self, img, o_size):
        ''' Replace trainIDs in prediction with colours from dict, reshape it afterwards to input dimensions and
            convert RGB to BGR to match openCV's colour system.
        '''
        sort_idx = np.argsort(self.id_color_keys)
        idx = np.searchsorted(self.id_color_keys, img, sorter=sort_idx)
        img = np.asarray(self.id_color_vals)[sort_idx][idx]
        img = img.astype(np.uint8)
        img = np.reshape(img, newshape=(o_size[0], o_size[1], 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img


    def _save_probs_to_disk(self, output):
        ''' Save the network output as numpy npy-file to disk. Only every self.opt.probs_frequency-th image is saved
            to disk!
        '''
        for i in range(output.shape[0]):
            if self.probs_idx % self.opt.probs_frequency == 0:
                np.save(os.path.join(self.logit_path, 'seg_logit_{}'.format(self.probs_idx)), output[i])

            self.probs_idx += 1


    def _print_options(self):
        ''' Print validation options to stdout
        '''
        # Convert namespace to dictionary
        opts = vars(self.opt)

        # Get max key length for left justifying
        max_len = max([len(key) for key in opts.keys()])

        # Print options to stdout
        print("+++++++++++++++++++++++++++ OPTIONS +++++++++++++++++++++++++++")
        for item in sorted(opts.items()):
            print(item[0].ljust(max_len), item[1])
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def _save_opts(self, n_eval):
        """Save options to disk so we know what we ran this experiment with
        """
        to_save = self.opt.__dict__.copy()
        to_save['n_eval'] = n_eval
        if self.opt.validate:
            filename = 'eval_opt.json'
        else:
            filename = 'eval_opt_{}.json'.format(self.opt.weights_epoch)

        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(to_save, f, indent=2)


if __name__ == "__main__":
    options = ERFnetOptions()
    opt = options.parse()
    evaluator = Evaluator(opt)
    evaluator.calculate_metrics()
