import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import torch
from torchvision.transforms import ToPILImage
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import mmcv
import pycocotools.mask as maskUtils
from mmdet.visualization.image import imshow_det_bboxes
import pickle

from dataloaders import make_data_loader
from dataloaders.utils import decode_seg_map_sequence, Colorize
from utils.metrics import Evaluator
from models.rfnet import RFNet
from models import rfnet_for_unseen
from models.resnet.resnet_single_scale_single_attention import *
from models.resnet import resnet_single_scale_single_attention_unseen
import torch.backends.cudnn as cudnn
from cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL
import torch.nn.functional as F
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

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

    def segformer_segmentation(self, image, processor, model, rank):
        h, w, _ = image.shape
        inputs = processor(images=image, return_tensors="pt").to(rank)
        outputs = model(**inputs)
        logits = outputs.logits
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        predicted_semantic_map = logits.argmax(dim=1)
        return predicted_semantic_map

    def draw_mask(self, image_name, mask, output_path):
        img = mmcv.imread(image_name)
        anns = {'annotations': mask}
        anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
        semantc_mask = torch.zeros(1024, 2048)
        i = 0
        for ann in anns['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            semantc_mask[valid_mask] = i
            i += 1
        sematic_class_in_img = torch.unique(semantc_mask)
        semantic_bitmasks, semantic_class_names = [], []

        # semantic prediction
        for i in range(len(sematic_class_in_img)):
            class_name = str(i)
            class_mask = semantc_mask == sematic_class_in_img[i]
            class_mask = class_mask.cpu().numpy().astype(np.uint8)
            semantic_class_names.append(class_name)
            semantic_bitmasks.append(class_mask)

        length = len(image_name)
        for i in range(length):
            if image_name[length-i-1] == '_':
                break
        filename = image_name[length-i:]
        imshow_det_bboxes(img,
                        bboxes=None,
                        labels=np.arange(len(sematic_class_in_img)),
                        segms=np.stack(semantic_bitmasks),
                        class_names=semantic_class_names,
                        font_size=25,
                        show=False,
                        out_file=os.path.join(output_path, filename + '_mask.png'))
        print('[Save] save mask: ', os.path.join(output_path, filename + '_mask.png'))
        semantc_mask = semantc_mask.unsqueeze(0).numpy()
        del img
        del semantic_bitmasks
        del semantic_class_names

    def draw_picture(self, image_name, semantc_mask, id2label, output_path, suffix):
        img = mmcv.imread(image_name)
        sematic_class_in_img = torch.unique(semantc_mask)
        semantic_bitmasks, semantic_class_names = [], []

        # semantic prediction
        for i in range(len(sematic_class_in_img)):
            class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
            class_mask = semantc_mask == sematic_class_in_img[i]
            class_mask = class_mask.cpu().numpy().astype(np.uint8)
            semantic_class_names.append(class_name)
            semantic_bitmasks.append(class_mask)

        #print(os.environ["OUTPUT_URL"])
        length = len(image_name)
        for i in range(length):
            if image_name[length-i-1] == '_':
                break
        filename = image_name[length-i:]
        imshow_det_bboxes(img,
                            bboxes=None,
                            labels=np.arange(len(sematic_class_in_img)),
                            segms=np.stack(semantic_bitmasks),
                            class_names=semantic_class_names,
                            font_size=25,
                            show=False,
                            out_file=os.path.join(output_path, filename + suffix))
        print('[Save] save rfnet prediction: ', os.path.join(output_path, filename + suffix))
        #semantc_mask = semantc_mask.unsqueeze(0).numpy()
        del img
        del semantic_bitmasks
        del semantic_class_names

    def confidence(self, input_output):
        output = torch.softmax(input_output, dim=0)
        highth = len(output[0])
        width = len(output[0][0])
        sum_1 = 0.0
        sum_2 = 0.0
        values, _ = torch.topk(output, k=2, dim=0)
        sum_1 = torch.sum(values[0])
        value_2 = torch.sub(values[0],values[1])
        sum_2 = torch.sum(value_2)
        sum_1 = sum_1/(highth*width)
        sum_2 = sum_2/(highth*width)
        count = (values[0] > 0.9).sum().item()
        sum_3 = count/(highth*width)
        return sum_3
    
    def sam_predict_ssa(self, image_name, pred):
        with open('/home/hsj/ianvs/project/cache.pickle', 'rb') as file:
            cache = pickle.load(file)
        img = mmcv.imread(image_name)
        if image_name in cache.keys():
            mask = cache[image_name]
            print("load cache")
        else:
            sam = sam_model_registry["vit_h"](checkpoint="/home/hsj/ianvs/project/segment-anything/sam_vit_h_4b8939.pth").to('cuda:1')
            mask_branch_model = SamAutomaticMaskGenerator(
                model=sam,
                #points_per_side=64,
                # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
                #pred_iou_thresh=0.86,
                #stability_score_thresh=0.92,
                #crop_n_layers=1,
                #crop_n_points_downscale_factor=2,
                #min_mask_region_area=100,  # Requires open-cv to run post-processing
                output_mode='coco_rle',
            )
            print('[Model loaded] Mask branch (SAM) is loaded.')
            mask = mask_branch_model.generate(img)
            cache[image_name] = mask
            with open('/home/hsj/ianvs/project/cache.pickle', 'wb') as file:
                pickle.dump(cache, file)
                print("save cache")

        anns = {'annotations': mask}
        #print(len(anns['annotations']), len(anns['annotations'][0]))
        #print(pred.shape)
        #print(pred[0])
        class_names = []
        semantc_mask = pred.clone()
        id2label = CONFIG_CITYSCAPES_ID2LABEL
        anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
        for ann in anns['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            #print(valid_mask)
            propose_classes_ids = pred[valid_mask]
            num_class_proposals = len(torch.unique(propose_classes_ids))
            if num_class_proposals == 1:
                semantc_mask[valid_mask] = propose_classes_ids[0]
                ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                class_names.append(ann['class_name'])
                # bitmasks.append(maskUtils.decode(ann['segmentation']))
                continue
            top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
            top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]
            #print("top_1_propose_class_ids: ", top_1_propose_class_ids)
            semantc_mask[valid_mask] = top_1_propose_class_ids
            ann['class_name'] = top_1_propose_class_names[0]
            ann['class_proposals'] = top_1_propose_class_names[0]
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))

            del valid_mask
            del propose_classes_ids
            del num_class_proposals
            del top_1_propose_class_ids
            del top_1_propose_class_names

        #print(semantc_mask.shape)
        #print(semantc_mask)
        
        del img
        del anns
        #del semantc_mask
        # del bitmasks
        del class_names
        return semantc_mask, mask

    def sam_predict(self, image_name, pred):
        with open('/home/hsj/ianvs/project/cache.pickle', 'rb') as file:
            cache = pickle.load(file)
        img = mmcv.imread(image_name)
        if image_name in cache.keys():
            mask = cache[image_name]
            print("load cache")
        else:
            sam = sam_model_registry["vit_h"](checkpoint="/home/hsj/ianvs/project/segment-anything/sam_vit_h_4b8939.pth").to('cuda:1')
            mask_branch_model = SamAutomaticMaskGenerator(
                model=sam,
                #points_per_side=64,
                # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
                #pred_iou_thresh=0.86,
                #stability_score_thresh=0.92,
                #crop_n_layers=1,
                #crop_n_points_downscale_factor=2,
                #min_mask_region_area=100,  # Requires open-cv to run post-processing
                output_mode='coco_rle',
            )
            print('[Model loaded] Mask branch (SAM) is loaded.')
            mask = mask_branch_model.generate(img)
            cache[image_name] = mask
            with open('/home/hsj/ianvs/project/cache.pickle', 'wb') as file:
                pickle.dump(cache, file)
                print("save cache")

        anns = {'annotations': mask}
        #print(len(anns['annotations']), len(anns['annotations'][0]))
        #print(pred.shape)
        #print(pred[0])
        class_names = []
        pred_2 = np.argmax(pred, axis=0)
        semantc_mask = pred_2.clone()
        id2label = CONFIG_CITYSCAPES_ID2LABEL
        anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
        for ann in anns['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            #print(valid_mask)
            length = len(pred)
            all_scores = [0 for i in range(length)]
            for i in range(length):
                propose_classes_ids = pred[i][valid_mask]
                #print(propose_classes_ids.shape)
                all_scores[i] = torch.sum(propose_classes_ids)
                #print(all_scores[i])
            top_1_propose_class_ids = np.argmax(all_scores)
            #print(top_1_propose_class_ids)
            top_1_propose_class_names = id2label['id2label'][str(top_1_propose_class_ids)]

            semantc_mask[valid_mask] = top_1_propose_class_ids
            ann['class_name'] = top_1_propose_class_names
            ann['class_proposals'] = top_1_propose_class_names
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))

            del valid_mask
            del propose_classes_ids
            del top_1_propose_class_ids
            del top_1_propose_class_names

        #print(semantc_mask.shape)
        #print(semantc_mask)

        #self.draw_picture(img, image_name, pred_2, id2label, output_path, "_origin.png")
        #self.draw_picture(img, image_name, semantc_mask, id2label, output_path, "_sam.png")

        del img
        del anns
        #del semantc_mask
        # del bitmasks
        del class_names
        return semantc_mask, mask

    def validate(self):
        #print("start validating 55")        
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        predictions = []
        scores = []
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

            if len(output) == 1:
                score = self.confidence(output[0])
            else:
                score = self.confidence(output)
            scores.append(score)
            
            pred = output.data.cpu().numpy()
            # todo
            pred = np.argmax(pred, axis=1)
            predictions.append(pred)

            output_path = os.environ["OUTPUT_URL"]
            id2label = CONFIG_CITYSCAPES_ID2LABEL
            self.draw_picture(image_name[0], torch.from_numpy(pred[0]), id2label, output_path, "_origin.png")
            
        #print("start validating 120")
        return predictions, scores
    
    def vit_validate(self):
        #print("start validating 55")        
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        predictions = []
        rank = 'cuda:0'
        semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(rank)
        for i, (sample, image_name) in enumerate(tbar):#self.test_loader:
            img = mmcv.imread(image_name[0])
            class_ids = self.segformer_segmentation(img, semantic_branch_processor, semantic_branch_model, rank)
            pred = class_ids.data.cpu().numpy()
            predictions.append(pred)

            output_path = os.environ["OUTPUT_URL"]
            id2label = CONFIG_CITYSCAPES_ID2LABEL
            self.draw_picture(image_name[0], torch.from_numpy(pred[0]), id2label, output_path, "_vit_origin.png")
            
        #print("start validating 120")
        return predictions

    def validate_cloud(self):
        #print("start validating 55")        
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        predictions = []
        scores = []
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

            if len(output) == 1:
                score = self.confidence(output[0])
            else:
                score = self.confidence(output)
            scores.append(score)
            
            pred = output.data.cpu().numpy()
            # todo
            pred_sam, mask = self.sam_predict(image_name[0], torch.from_numpy(pred[0]))
            if pred_sam.ndim < 3:
                h, w = pred_sam.shape
                pred_sam = pred_sam.reshape(1, h, w)
            #print(pred_sam.shape)

            predictions.append(np.array(pred_sam))

            output_path = os.environ["OUTPUT_URL"]
            id2label = CONFIG_CITYSCAPES_ID2LABEL
            self.draw_picture(image_name[0], pred_sam[0], id2label, output_path, "_sam.png")
            self.draw_mask(image_name[0], mask, output_path)
            
        #print("start validating 120")
        return predictions, scores

    def vit_validate_cloud(self):
        #print("start validating 55")        
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        predictions = []
        rank = 'cuda:0'
        semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(rank)
        for i, (sample, image_name) in enumerate(tbar):#self.test_loader:
            img = mmcv.imread(image_name[0])
            class_ids = self.segformer_segmentation(img, semantic_branch_processor, semantic_branch_model, rank)
            pred = class_ids.data.cpu().numpy()
            pred_sam, mask = self.sam_predict_ssa(image_name[0], torch.from_numpy(pred[0]))
            if pred_sam.ndim < 3:
                h, w = pred_sam.shape
                pred_sam = pred_sam.reshape(1, h, w)
            #print(pred_sam.shape)
            predictions.append(np.array(pred_sam))

            output_path = os.environ["OUTPUT_URL"]
            id2label = CONFIG_CITYSCAPES_ID2LABEL
            self.draw_picture(image_name[0], torch.from_numpy(pred[0]), id2label, output_path, "_vit_origin.png")
            self.draw_picture(image_name[0], pred_sam[0], id2label, output_path, "_vit_sam.png")
            self.draw_mask(image_name[0], mask, output_path)
            
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
