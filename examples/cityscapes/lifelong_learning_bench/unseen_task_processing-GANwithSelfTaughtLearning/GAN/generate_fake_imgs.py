import torch

from models import Generator, weights_init

import matplotlib.pyplot as plt

import os

from collections import OrderedDict

import numpy as np

from skimage import io

import sys
# Resolve paths relative to this file so the script is directory-agnostic, and
# ensure the example root is importable so `from util import load_yaml` works
# when this script is run from inside the GAN/ directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..")))
from util import load_yaml


device = 'cuda'

ngf = 64
nz = 256
im_size = 1024
netG = Generator(ngf=ngf, nz=nz, im_size=im_size).to(device)
weights_init(netG)
# BUG-3: read the GAN checkpoint name/iter from config instead of hardcoding.
configs = load_yaml(os.path.join(BASE_DIR, '..', 'config.yaml'))
# Flatten the list-of-single-key-dicts section so lookups are order-independent.
gan_cfg = {k: v for entry in configs['GAN'] for k, v in entry.items()}
gan_name = gan_cfg.get('name')
gan_iter = gan_cfg.get('iter')
weights = torch.load(os.path.join(BASE_DIR, 'train_results', gan_name, 'models', f'{gan_iter}.pth'))
netG_weights = OrderedDict()
for name, weight in weights['g'].items():
    name = name.split('.')[1:]
    name = '.'.join(name)
    netG_weights[name] = weight
netG.load_state_dict(netG_weights)
current_batch_size = 1


index = 1
while index <= 3000:
    noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
    fake_images = netG(noise)[0]
    for fake_image in fake_images:
        fake_image = fake_image.detach().cpu().numpy().transpose(1, 2, 0)
        fake_image = fake_image * np.array([0.5, 0.5, 0.5])
        fake_image = fake_image + np.array([0.5, 0.5, 0.5])
        fake_image = (fake_image * 255).astype(np.uint8)
        io.imsave('../data/fake_imgs/' + str(index) + '.png', fake_image)
        index += 1
