# @inproceedings{zhou2021learning,
# author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
# title = {Learning Placeholders for Open-Set Recognition},
# booktitle = {CVPR},
# pages = {4401-4410},
# year = {2021}
# }
from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

__all__ = ('UnseenSampleRecognitionByScene',)

@ClassFactory.register(ClassType.UTD, alias="UnseenSampleRecognitionByScene")
class UnseenSampleRecognitionByScene:
    """
    Dividing the data set based on whether the sample is a known class of data.

    Parameters
    ----------
    model_path: string
        Path of the model
    """
    def __init__(self, **kwargs):
        self.model_path = kwargs.get("model_path")

    def __call__(self,
                 samples: BaseDataSource, **kwargs):
        from torch.utils.data import DataLoader
        import torch
        import torchvision
        from torchvision import transforms
        from PIL import Image
        from models.wide_resnet_embedding import Wide_ResNet
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y
        data_transforms = transforms.Compose([
                    transforms.Resize([32,32]),
                    transforms.ToTensor()
                    ])
        InferenceData=data_transforms(Image.open(x_data[0][0]))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pthfile = self.model_path
        # Instantiate the scene-recognition model and load the trained weights
        # into it. torch.load() returns only a state_dict (an OrderedDict),
        # which is not callable; it must be loaded into a model instance before
        # running inference. The architecture (WRN-28-10) and the binary
        # seen/unseen head (num_classes=2) match the downstream logic below.
        model = Wide_ResNet(28, 10, 0.3, 2).to(device)
        model.load_state_dict(torch.load(pthfile, map_location=device))
        model.eval()
        unseen_image = BaseDataSource(data_type=d_type)
        unseen_image.x, unseen_image.y = [], []
        seen_image = BaseDataSource(data_type=d_type)
        seen_image.x, seen_image.y = [], []
        InferenceData = InferenceData.to(device)
        InferenceData = torch.unsqueeze(InferenceData, 0)
        with torch.no_grad():
            outputs = model(InferenceData)
        a, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy()
        if predicted[0] == 1:
            seen_image.x.append(x_data[0])
        if predicted[0] == 0:
            unseen_image.x.append(x_data[0])
        
        return seen_image, unseen_image
