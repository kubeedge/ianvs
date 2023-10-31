import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import copy

from mypath import Path
from dataloaders import make_data_loader

from models.erfnet_RA_parallel import Net as Net_RAP

from utils.loss import SegmentationLosses
from models.replicate import patch_replication_callback
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from sedna.datasources import BaseDataSource

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"

class Trainer(object):
    def __init__(self, args, train_data=None, valid_data=None):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # denormalize for detph image
        self.mean_depth = torch.as_tensor(0.12176, dtype=torch.float32, device='cpu')
        self.std_depth = torch.as_tensor(0.09752, dtype=torch.float32, device='cpu')

        self.nclass = args.num_class # [13, 30, 30]
        self.current_domain = min(args.current_domain, 2) # current domain start from 0 and maximum is 2
        self.next_domain = args.next_domain # next_domain start from 1

        if self.current_domain <= 0:
            self.current_class = [self.nclass[0]]
        elif self.current_domain == 1:
            self.current_class = self.nclass[:2]
        elif self.current_domain >= 2:
            self.current_class = self.nclass
        else:
            pass

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, _ = make_data_loader(args, train_data=train_data, 
                                                                                   valid_data=valid_data, **kwargs)   

        self.print_domain_info()

        # Define network
        model = Net_RAP(num_classes=self.current_class, nb_tasks=self.current_domain + 1, cur_task=self.current_domain)
        model_old = Net_RAP(num_classes=self.current_class, nb_tasks=self.current_domain, cur_task=max(self.current_domain-1, 0))
        args.current_domain = self.next_domain
        args.next_domain += 1

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # Define loss function
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, gpu_ids=args.gpu_ids).build_loss(mode=args.loss_type)
        self.model, self.model_old, self.optimizer = model, model_old, optimizer
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass[self.current_domain])
        # # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            # patch_replication_callback(self.model)
            self.model = self.model.cuda(args.gpu_ids)
        self.gpu_ids = args.gpu_ids
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            print(f"Training: load model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=torch.device('cuda:0'))
            args.start_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['state_dict'], False)

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def get_weight(self):
        print("get weight")
        current_model = copy.deepcopy(self.model)
        return current_model.parameters()
    
    def set_weight(self, weights):
        length = len(weights)
        print("set weight", length)
        print("model:", self.args.resume)
        tau = 0.2
        if length == 1:  
            for param, target_param in zip(weights[0], self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        elif length == 2:
            for param1, param2, target_param in zip(weights[0], weights[1], self.model.parameters()):
                target_param.data.copy_(0.5 * tau * param1.data + 0.5 * tau * param2.data + (1 - tau) * target_param.data)

    def my_training(self, epoch):
        train_loss = 0.0
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        current_model = copy.deepcopy(self.model)
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
                #print(target.shape)
            else:
                image, target = sample['image'], sample['label']
                print(image.shape)
            if self.args.cuda:
                image, target = image.cuda(self.args.gpu_ids), target.cuda(self.args.gpu_ids)
                if self.args.depth:
                    depth = depth.cuda(self.args.gpu_ids)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            if self.args.depth:
                output = self.model(image, depth)
            else:
                output = self.model(image)
            target[target > self.nclass[2]-1] = 255
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            #print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10 + 1) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.depth:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

                    depth_display = depth[0].cpu().unsqueeze(0)
                    depth_display = depth_display.mul_(self.std_depth).add_(self.mean_depth)
                    depth_display = depth_display.numpy()
                    depth_display = depth_display*255
                    depth_display = depth_display.astype(np.uint8)
                    self.writer.add_image('Depth', depth_display, global_step)

                else:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        tau = 0.3
        flag = True
        for param, target_param in zip(current_model.parameters(), self.model.parameters()):
            if flag:
                flag = False
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        del current_model
        return train_loss

    def training(self, epoch):
        train_loss = 0.0
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])

        self.model.train()
        self.model_old.eval()

        for name, m in self.model_old.named_parameters():
            m.requires_grad = False

        for name, m in self.model.named_parameters():
            if 'decoder' in name:
                if 'decoder.{}'.format(self.current_domain) in name:
                    m.requires_grad = True
                else:
                    m.requires_grad = False

            elif 'encoder' in name:
                if 'bn' in name or 'parallel_conv' in name:
                    if '.{}.weight'.format(self.current_domain) in name or '.{}.bias'.format(self.current_domain) in name:
                        m.requires_grad = True
                    else:
                        m.requires_grad = False

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
                #print(target.shape)
            else:
                image, target = sample['image'], sample['label']
                # print(image.shape)
            if self.args.cuda:
                image, target = image.cuda(self.args.gpu_ids), target.cuda(self.args.gpu_ids)
                if self.args.depth:
                    depth = depth.cuda(self.args.gpu_ids)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            if self.args.depth:
                output = self.model(image, depth)
            else:
                output = self.model(image, self.current_domain)

            output = torch.tensor(output, dtype=torch.float32) 
            target[target > self.nclass[self.current_domain]-1] = 255

            target = self.my_to_label(target)
            target = self.my_relabel(target, 255, self.nclass[self.current_domain] - 1)

            target = target.squeeze(0)
            target = target.cuda(self.gpu_ids)

            outputs_prev_task = self.model(image, max(self.current_domain-1, 0))
            loss = self.criterion(output, target) 

            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10 + 1) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.depth:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

                    depth_display = depth[0].cpu().unsqueeze(0)
                    depth_display = depth_display.mul_(self.std_depth).add_(self.mean_depth)
                    depth_display = depth_display.numpy()
                    depth_display = depth_display*255
                    depth_display = depth_display.astype(np.uint8)
                    self.writer.add_image('Depth', depth_display, global_step)

                else:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # save checkpoint every epoch
        checkpoint_path = self.saver.save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'best_pred': self.best_pred,
                            }, True)
        return train_loss

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, (sample, img_path) in enumerate(tbar):
            if self.args.depth:
                image, depth, target = sample['image'], sample['depth'], sample['label']
            else:
                image, target = sample['image'], sample['label']
                # print(f"val image is {image}")
            if self.args.cuda:
                image, target = image.cuda(self.args.gpu_ids), target.cuda(self.args.gpu_ids)
                if self.args.depth:
                    depth = depth.cuda(self.args.gpu_ids)
            with torch.no_grad():
                if self.args.depth:
                    output = self.model(image, depth)
                else:
                    output = self.model(image)
            target[target > self.nclass[2]-1] = 255
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def print_domain_info(self):
       
        domain_map = {
            0: "Synthia",
            1: "CityScapes",
            2: "Cloud-Robotics"
        }

        domain_name = domain_map.get(self.current_domain, "Unknown Domain")

        print("We are in domain", self.current_domain, "which is", domain_name)

    def my_relabel(self, tensor, olabel, nlabel):
        tensor[tensor == olabel] = nlabel
        return tensor

    def my_to_label(self, image):
        image = image.cpu()
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)

