from sedna.common.log import LOGGER
from sedna.algorithms.unseen_task_processing.GANwithSelfTaughtLearning.util import load_yaml
import matplotlib.pyplot as plt
import os

from sedna.algorithms.unseen_task_processing.GANwithSelfTaughtLearning.deeplabv3.datasets import DatasetTrain, DatasetVal
from sedna.algorithms.unseen_task_processing.GANwithSelfTaughtLearning.deeplabv3.model.deeplabv3 import DeepLabV3
import sys
from sedna.algorithms.unseen_task_processing.GANwithSelfTaughtLearning.deeplabv3.utils.utils import add_weight_decay

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.enc3 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        return x


def _flatten_config(entries):
    """Merge a list of single-key dicts (as used in config.yaml) into one dict.

    This makes config lookups robust to ordering changes in config.yaml.
    """
    flat = {}
    for entry in entries:
        flat.update(entry)
    return flat


def train_deepblabv3():
    # Resolve paths relative to this file so the script is directory-agnostic.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs = load_yaml(os.path.join(base_dir, '..', 'config.yaml'))

    # Flatten the list-of-single-key-dicts config sections into plain dicts so
    # lookups are robust to ordering changes in config.yaml.
    deeplabv3_cfg = _flatten_config(configs['deeplabv3'])
    stl_cfg = _flatten_config(configs['STL'])

    # BUG-1: validate required config paths up front so a missing value fails
    # immediately with a clear, named error instead of a cryptic downstream crash.
    required_paths = {
        'cityscapes_data_path': deeplabv3_cfg.get('cityscapes_data_path'),
        'cityscapes_meta_path': deeplabv3_cfg.get('cityscapes_meta_path'),
        'class_weights': deeplabv3_cfg.get('class_weights'),
    }
    for key, value in required_paths.items():
        if not value:
            raise ValueError(
                f"config.yaml: '{key}' is empty. Please set this path.")

    model_id = deeplabv3_cfg.get('name')

    # BUG-2: build the encoder checkpoint path from config (STL name/iter)
    # instead of a hardcoded developer path.
    stl_name = stl_cfg.get('name')
    stl_epochs = stl_cfg.get('iter')
    encoder = Encoder().cuda()
    encoder_path = os.path.join(
        base_dir, '..', 'selftaughtlearning', 'train_results',
        stl_name, f'encoder{stl_epochs}.pth')
    encoder.load_state_dict(torch.load(encoder_path))
    # The encoder is a fixed preprocessing transform: switch it to eval mode and
    # freeze its parameters so it is not updated during DeepLabV3 training.
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    num_epochs = deeplabv3_cfg.get('iter')
    batch_size = deeplabv3_cfg.get('batch_size')
    learning_rate = deeplabv3_cfg.get('lr')

    network = DeepLabV3(model_id, project_dir=os.getcwd()).cuda()

    train_dataset = DatasetTrain(cityscapes_data_path=deeplabv3_cfg.get('cityscapes_data_path'),
                                 cityscapes_meta_path=deeplabv3_cfg.get('cityscapes_meta_path'))
    val_dataset = DatasetVal(cityscapes_data_path=deeplabv3_cfg.get('cityscapes_data_path'),
                             cityscapes_meta_path=deeplabv3_cfg.get('cityscapes_meta_path'))

    num_train_batches = int(len(train_dataset) / batch_size)
    num_val_batches = int(len(val_dataset) / batch_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=1)

    params = add_weight_decay(network, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    with open(deeplabv3_cfg.get('class_weights'), "rb") as file:
        class_weights = np.array(pickle.load(file))
    class_weights = torch.from_numpy(class_weights)
    class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    epoch_losses_train = []
    epoch_losses_val = []
    for epoch in range(num_epochs):
        LOGGER.info("epoch: %d/%d" % (epoch + 1, num_epochs))

        ############################################################################
        # train:
        ############################################################################
        network.train()  # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs) in enumerate(train_loader):
            imgs = Variable(imgs).cuda()
            # encoder images
            imgs = encoder(imgs)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()

            outputs = network(imgs)

            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        LOGGER.info("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
        plt.close(1)

        network.eval()
        batch_losses = []
        for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
            with torch.no_grad():
                imgs = Variable(imgs).cuda()
                # encoder images - match training distribution (BUG-5)
                imgs = encoder(imgs)
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()

                outputs = network(imgs)
                loss = loss_fn(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_val, file)
        LOGGER.info("val loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
        plt.close(1)


if __name__ == '__main__':
    train_deepblabv3()
