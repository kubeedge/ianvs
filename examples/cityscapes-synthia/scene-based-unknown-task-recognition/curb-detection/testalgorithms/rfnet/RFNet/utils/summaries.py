<<<<<<< HEAD
import os
import torch
from torchvision.utils import make_grid
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step, depth=None):
        if depth is None:
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Image', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, global_step)
        else:
            grid_image = make_grid(image[:3].clone().cpu().data, 4, normalize=True)
            writer.add_image('Image', grid_image, global_step)

            grid_image = make_grid(depth[:3].clone().cpu().data, 4, normalize=True)  # normalize=False?
            writer.add_image('Depth', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 4, normalize=False, range=(0, 255))
            writer.add_image('Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 4, normalize=False, range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, global_step)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:4c4a6b5f7da2045d096b6161a580ebd4282bb6f8e07342a0e9b3f3f28d6cf9d0
size 2229
>>>>>>> 9676c3e (ya toh aar ya toh par)
