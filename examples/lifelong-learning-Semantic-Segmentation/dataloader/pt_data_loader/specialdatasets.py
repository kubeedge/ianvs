# MIT License
#
# Copyright (c) 2020 Marvin Klingner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader.pt_data_loader.basedataset import BaseDataset
import dataloader.pt_data_loader.mytransforms as mytransforms
import dataloader.definitions.labels_file as lf


class StandardDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(StandardDataset, self).__init__(*args, **kwargs)

        if self.disable_const_items is False:
            assert self.parameters.K is not None and self.parameters.stereo_T is not None, '''There are no K matrix and
            stereo_T parameter available for this dataset.'''

    def add_const_dataset_items(self, sample):
        K = self.parameters.K.copy()

        native_key = ('color', 0, -1) if (('color', 0, -1) in sample) else ('color_right', 0, -1)
        native_im_shape = sample[native_key].shape

        K[0, :] *= native_im_shape[1]
        K[1, :] *= native_im_shape[0]

        sample["K", -1] = K
        sample["stereo_T"] = self.parameters.stereo_T

        return sample


class KITTIDataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)


class MapillaryDataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(MapillaryDataset, self).__init__(*args, **kwargs)


class CityscapesDataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)


class Gta5Dataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(Gta5Dataset, self).__init__(*args, **kwargs)


class SimpleDataset(BaseDataset):
    '''
        Dataset that uses the Simple Mode. keys_to_load must be specified.
    '''
    def __init__(self, *args, **kwargs):
        super(SimpleDataset, self).__init__(*args, **kwargs)

    def add_const_dataset_items(self, sample):
        return sample


if __name__ == '__main__':
    """
    The following code is an example of how a dataloader object can be created for a specific dataset. In this case,
    the cityscapes dataset is used. 
    
    Every dataset should be created using the StandardDataset class. Necessary arguments for its constructor are the
    dataset name and the information whether to load the train, validation or test split. In this standard setting,
    for every dataset only the color images are loaded. The user can pass a list of keys_to_load in order to also use
    other data categories, depending on what is available in the dataset. It is also possible to define a list of
    transforms that are performed every time an image is loaded from the dataset.
    """
    def print_dataset(dataloader, num_elements=3):
        """
        This little function prints the size of every element in a certain amount of dataloader samples.

        :param dataloader: dataloader object that yields the samples
        :param num_elements: number of samples of which the sizes are to be printed
        """
        for element, i in zip(dataloader, range(num_elements)):
            print('+++ Image {} +++'.format(i))
            for key in element.keys():
                print(key, element[key].shape)
            plt.imshow(np.array(element[('color', 0, 0)])[0, :, :, :].transpose(1, 2, 0))

    # Simple example of how to load a dataset. Every supported dataset can be loaded that way.
    dataset = 'cityscapes'
    trainvaltest_split = 'train'
    keys_to_load = ['color', 'depth', 'segmentation', 'camera_intrinsics']   # Optional; standard is just 'color'

    # The following parametes and the data_transforms list are optional. Standard is just the transform ToTensor()
    width = 640
    height = 192
    scales = [0, 1, 2, 3]
    data_transforms = [#mytransforms.RandomExchangeStereo(),  # (color, 0, -1)
                       mytransforms.RandomHorizontalFlip(),
                       mytransforms.RandomVerticalFlip(),
                       mytransforms.CreateScaledImage(),  # (color, 0, 0)
                       mytransforms.RandomRotate(0.0),
                       mytransforms.RandomTranslate(0),
                       mytransforms.RandomRescale(scale=1.1, fraction=0.5),
                       mytransforms.RandomCrop((320, 1088)),
                       mytransforms.Resize((height, width)),
                       mytransforms.MultiResize(scales),
                       mytransforms.CreateColoraug(new_element=True, scales=scales),  # (color_aug, 0, 0)
                       mytransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                hue=0.1, gamma=0.0),
                       mytransforms.GaussianBlurr(fraction=0.5),
                       mytransforms.RemoveOriginals(),
                       mytransforms.ToTensor(),
                       mytransforms.NormalizeZeroMean(),
                       ]

    print('Loading {} dataset, {} split'.format(dataset, trainvaltest_split))
    traindataset = StandardDataset(dataset,
                                   trainvaltest_split,
                                   keys_to_load=keys_to_load,
                                   stereo_mode='mono',
                                   keys_to_stereo=['color', 'depth', 'segmentation'],
                                   data_transforms=data_transforms
                                   )
    trainloader = DataLoader(traindataset, batch_size=1,
                             shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    print(traindataset.stereo_mode)
    print_dataset(trainloader)