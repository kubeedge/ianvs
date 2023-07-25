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
from models.rfnet import RFNet
from models import rfnet_for_unseen
from models.resnet.resnet_single_scale_single_attention import *
from models.resnet import resnet_single_scale_single_attention_unseen
import torch.backends.cudnn as cudnn

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"

class Validator(object):
    def __init__(self, args, data=None, unseen_detection=False):
        self.args = args
        self.time_train = []
        self.num_class = args.num_class

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        # _, self.val_loader, _, self.custom_loader, self.num_class = make_data_loader(args, **kwargs)
        _, _, self.test_loader, _ = make_data_loader(args, test_data=data, **kwargs)
        print('un_classes:'+str(self.num_class))

        # Define evaluator
        self.evaluator = Evaluator(self.num_class)

        # Define network
        if unseen_detection:
            self.resnet = resnet_single_scale_single_attention_unseen.\
                resnet18(pretrained=False, efficient=False, use_bn=True)
            self.model = rfnet_for_unseen.RFNet(self.resnet, num_classes=self.num_class, use_bn=True)
        else:
            self.resnet = resnet18(pretrained=False, efficient=False, use_bn=True)
            self.model = RFNet(self.resnet, num_classes=self.num_class, use_bn=True)

        if args.cuda:
            #self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda(args.gpu_ids)
            cudnn.benchmark = True  # accelarate speed
        print('Model loaded successfully!')

        # # Load weights
        # assert os.path.exists(args.weight_path), 'weight-path:{} doesn\'t exit!'.format(args.weight_path)
        # self.new_state_dict = torch.load(args.weight_path, map_location=torch.device("cpu"))
        # self.model = load_my_state_dict(self.model, self.new_state_dict['state_dict'])

    def validate(self):
        #print("start validating 55")        
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        predictions = []
        for i, (sample, image_name) in enumerate(tbar):#self.test_loader:
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
            else:
                # spec = time.time()
                image, target = sample['image'], sample['label']            
            #print(self.args.cuda, self.args.gpu_ids)
            if self.args.cuda:
                image = image.cuda(self.args.gpu_ids)
                if self.args.depth:
                    depth = depth.cuda(self.args.gpu_ids)
                    
            with torch.no_grad():
                if self.args.depth:
                    output = self.model(image, depth)
                else:
                    output = self.model(image)
                    
            if self.args.cuda:
                torch.cuda.synchronize()

            pred = output.data.cpu().numpy()
            # todo
            pred = np.argmax(pred, axis=1)
            predictions.append(pred)

            if not self.args.save_predicted_image:
                continue
            
            pre_colors = Colorize()(torch.max(output, 1)[1].detach().cpu().byte())
            pre_labels = torch.max(output, 1)[1].detach().cpu().byte()
            print(pre_labels.shape)
            # save
            for i in range(pre_colors.shape[0]):
                print(image_name[0])

                if not image_name[0]:
                    img_name = "test.png"
                else:
                    img_name = os.path.basename(image_name[0])

                color_label_name = os.path.join(self.args.color_label_save_path, img_name)
                label_name = os.path.join(self.args.label_save_path, img_name)
                merge_label_name = os.path.join(self.args.merge_label_save_path, img_name)

                os.makedirs(os.path.dirname(color_label_name), exist_ok=True)
                os.makedirs(os.path.dirname(merge_label_name), exist_ok=True)
                os.makedirs(os.path.dirname(label_name), exist_ok=True)

                pre_color_image = ToPILImage()(pre_colors[i])  # pre_colors.dtype = float64
                pre_color_image.save(color_label_name)

                pre_label_image = ToPILImage()(pre_labels[i])
                pre_label_image.save(label_name)

                if (self.args.merge):
                    image_merge(image[i], pre_color_image, merge_label_name)
                    print('save image: {}'.format(merge_label_name))
        #print("start validating 120")
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
            pre_colors = Colorize()(torch.max(output, 1)[1].detach().cpu().byte())
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

def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('{} not in model_state'.format(name))
            continue
        else:
            own_state[name].copy_(param)

    return model
