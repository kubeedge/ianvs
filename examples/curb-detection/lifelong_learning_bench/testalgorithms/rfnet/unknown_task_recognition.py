from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

__all__ = ('SampleRegonitionByScene',)
@ClassFactory.register(ClassType.UTB, alias="SampleRegonitionByScene")
class SampleRegonitionByScene:
    def __init__(self, **kwargs):
        self.model_path = kwargs.get("model_path")
        self.path_inference_dataset = kwargs.get("path_inference_dataset")
        self.path_inference_sample = kwargs.get("path_inference_sample")
        self.path_seen_sample = kwargs.get("path_seen_sample")
        self.path_unseen_sample = kwargs.get("path_unseen_sample")
    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        from torch.utils.data import DataLoader
        import torch
        import torchvision
        from torchvision import transforms
        import os
        import shutil
        data_transforms = transforms.Compose([
                            transforms.Resize([32,32]),
                            transforms.ToTensor()
                            ])
        InferenceDataset = torchvision.datasets.ImageFolder(root=kwargs.path_inference_dataset,transform=data_transforms)
        inference_loader=DataLoader(InferenceDataset, batch_size=100, shuffle=True, num_workers=4)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pthfile = kwargs.model_path
        state_dict = torch.load(pthfile)
        unseen_image = []
        seen_image = []
        for batch_idx, (inputs, targets) in enumerate(inference_loader):
            inputs = inputs.to(device)
            outputs = state_dict(inputs)
            a, predicted = outputs.max(1)
            predicted = predicted.cpu().numpy()
            for i in range(len(predicted)):
                if predicted[i] == 1:
                    seen_image.append(i+batch_idx*100)
                if predicted[i] == 0:
                    unseen_image.append(i+batch_idx*100)
        filelist = os.listdir(kwargs.path_inference_sample)
        for i, file in enumerate(filelist):
            if i in seen_image:
                src = os.path.join(kwargs.inference_path, file)
                drt = os.path.join(kwargs.path_seen_sample, file)
                shutil.copy(src, drt)
            if i in unseen_image:
                src = os.path.join(kwargs.inference_path, file)
                drt = os.path.join(kwargs.path_unseen_sample, file)
                shutil.copy(src, drt)
        
        return kwargs.path_seen_image, kwargs.unseen_image
