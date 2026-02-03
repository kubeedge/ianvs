import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import torch
from torchvision.transforms import ToPILImage
from PIL import Image

from dataloaders import make_data_loader
from dataloaders.utils import decode_seg_map_sequence, Colorize
from utils.metrics import Evaluator
from models.erfnet_RA_parallel import Net as Net_RAP
import torch.backends.cudnn as cudnn

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class Validator(object):
    def __init__(self, args, data=None, unseen_detection=False):
        self.args = args
        self.time_train = []
        self.num_class = args.num_class  # [13, 30, 30]
        self.current_domain = args.current_domain  # 0 when start
        self.next_domain = args.next_domain  # 1 when start

        if self.current_domain <= 0:
            self.current_class = [self.num_class[0]]
        elif self.current_domain == 1:
            self.current_class = self.num_class[:2]
        elif self.current_domain >= 2:
            self.current_class = self.num_class
        else:
            pass

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        _, _, self.test_loader, _ = make_data_loader(
            args, test_data=data, **kwargs)

        # Define evaluator
        self.evaluator = Evaluator(self.num_class[self.current_domain])

        # Define network
        self.model = Net_RAP(num_classes=self.current_class,
                             nb_tasks=self.current_domain + 1, cur_task=self.current_domain)

        args.current_domain = self.next_domain
        args.next_domain += 1
        if args.cuda:
            self.model = self.model.cuda(args.gpu_ids)
            cudnn.benchmark = True  # accelerate speed
        print('Model loaded successfully!')

    def validate(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        predictions = []
        for i, (sample, image_name) in enumerate(tbar):
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
            else:
                image, target = sample['image'], sample['label']

            if self.args.cuda:
                image = image.cuda(self.args.gpu_ids)
                if self.args.depth:
                    depth = depth.cuda(self.args.gpu_ids)

            with torch.no_grad():
                if self.args.depth:
                    output = self.model(image, depth)
                else:
                    output = self.model(image, self.current_domain)

            if self.args.cuda:
                torch.cuda.synchronize()

            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            predictions.append(pred)

            if not self.args.save_predicted_image:
                continue

            pre_colors = Colorize()(torch.max(output, 1)[
                1].detach().cpu().byte())
            pre_labels = torch.max(output, 1)[1].detach().cpu().byte()
            print(pre_labels.shape)
            # save
            for i in range(pre_colors.shape[0]):
                print(image_name[0])

                if not image_name[0]:
                    img_name = "test.png"
                else:
                    img_name = os.path.basename(image_name[0])

                color_label_name = os.path.join(
                    self.args.color_label_save_path, img_name)
                label_name = os.path.join(self.args.label_save_path, img_name)
                merge_label_name = os.path.join(
                    self.args.merge_label_save_path, img_name)

                os.makedirs(os.path.dirname(color_label_name), exist_ok=True)
                os.makedirs(os.path.dirname(merge_label_name), exist_ok=True)
                os.makedirs(os.path.dirname(label_name), exist_ok=True)

                pre_color_image = ToPILImage()(pre_colors[i])
                pre_color_image.save(color_label_name)

                pre_label_image = ToPILImage()(pre_labels[i])
                pre_label_image.save(label_name)

                if (self.args.merge):
                    image_merge(image[i], pre_color_image, merge_label_name)
                    print('save image: {}'.format(merge_label_name))

        return predictions

    def task_divide(self):
        seen_task_samples, unseen_task_samples = [], []
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, (sample, image_name) in enumerate(tbar):

            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
            else:
                image, target = sample['image'], sample['label']
            if self.args.cuda:
                image = image.cuda(self.args.gpu_ids)
                if self.args.depth:
                    depth = depth.cuda(self.args.gpu_ids)
            start_time = time.time()
            with torch.no_grad():
                if self.args.depth:
                    output_, output, _ = self.model(image, depth)
                else:
                    output_, output, _ = self.model(image)
            if self.args.cuda:
                torch.cuda.synchronize()
            if i != 0:
                fwt = time.time() - start_time
                self.time_train.append(fwt)
                print("Forward time per img (bath size=%d): %.3f (Mean: %.3f)" % (
                    self.args.val_batch_size, fwt / self.args.val_batch_size,
                    sum(self.time_train) / len(self.time_train) / self.args.val_batch_size))
            time.sleep(0.1)  # to avoid overheating the GPU too much

            # pred colorize
            pre_colors = Colorize()(torch.max(output, 1)[
                1].detach().cpu().byte())
            pre_labels = torch.max(output, 1)[1].detach().cpu().byte()
            for i in range(pre_colors.shape[0]):
                task_sample = dict()
                task_sample.update(image=sample["image"][i])
                task_sample.update(label=sample["label"][i])
                if self.args.depth:
                    task_sample.update(depth=sample["depth"][i])

                if torch.max(pre_labels) == output.shape[1] - 1:
                    unseen_task_samples.append((task_sample, image_name[i]))
                else:
                    seen_task_samples.append((task_sample, image_name[i]))

        return seen_task_samples, unseen_task_samples


def image_merge(image, label, save_name):
    image = ToPILImage()(image.detach().cpu().byte())
    # width, height = image.size
    left = 140
    top = 30
    right = 2030
    bottom = 900
    # crop
    image = image.crop((left, top, right, bottom))
    # resize
    image = image.resize(label.size, Image.BILINEAR)

    image = image.convert('RGBA')
    label = label.convert('RGBA')
    image = Image.blend(image, label, 0.6)
    image.save(save_name)


# custom function to load model when not all dict elements
def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        else:
            own_state[name].copy_(param)

    return model
