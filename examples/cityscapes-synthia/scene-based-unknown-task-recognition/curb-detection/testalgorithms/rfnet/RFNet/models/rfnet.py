<<<<<<< HEAD
import torch.nn as nn
from itertools import chain # 串联多个迭代对象

from .util import _BNReluConv, upsample


class RFNet(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(RFNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        print(self.backbone.num_features)
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, rgb_inputs, depth_inputs = None):
        x, additional = self.backbone(rgb_inputs, depth_inputs)
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        #print(logits_upsample.size)
        return logits_upsample


    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
=======
version https://git-lfs.github.com/spec/v1
oid sha256:5976fd206efcac73bb59a3cc70e78006aa75ed79d3242dc286602379a26e950e
size 954
>>>>>>> 9676c3e (ya toh aar ya toh par)
