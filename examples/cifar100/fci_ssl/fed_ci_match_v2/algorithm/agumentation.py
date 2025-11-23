<<<<<<< HEAD
# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps


"""
Reference: https://github.com/heartInsert/randaugment
"""


class Rand_Augment:
    def __init__(self, Numbers=None, max_Magnitude=None):
        self.transforms = [
            "autocontrast",
            "equalize",
            "rotate",
            "solarize",
            "color",
            "posterize",
            "contrast",
            "brightness",
            "sharpness",
            "shearX",
            "shearY",
            "translateX",
            "translateY",
        ]
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 231, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.3, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fill=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fill=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor,
            ),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

    def rand_augment(self):
        """Generate a set of distortions.
        Args:
        N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
        M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude
        """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        operations = self.rand_augment()
        for op_name, M in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(
            rot, Image.new("RGBA", rot.size, (128,) * 4), rot
        ).convert(img.mode)

    def test_single_operation(self, image, op_name, M=-1):
        """
        :param image: image
        :param op_name: operation name in   self.transforms
        :param M: -1  stands  for the  max   Magnitude  in  there operation
        :return:
        """
        operation = self.func[op_name]
        mag = self.ranges[op_name][M]
        image = operation(image, mag)
        return image


class Base_Augment:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def __call__(self, images):
        return images


class Weak_Augment(Base_Augment):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)
        self.augment_impl = self.augment_for_cifar

    def augment_mirror(self, x):
        new_images = x.copy()
        indices = np.arange(len(new_images)).tolist()
        sampled = random.sample(
            indices, int(round(0.5 * len(indices)))
        )  # flip horizontally 50%
        new_images[sampled] = np.fliplr(new_images[sampled])
        return new_images  # random shift

    def augment_shift(self, x, w):
        y = tf.pad(x, [[0] * 2, [w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
        return tf.image.random_crop(y, tf.shape(x))

    def augment_for_cifar(self, images: np.ndarray):
        return self.augment_shift(self.augment_mirror(images), 4)

    def __call__(self, images: np.ndarray):
        return self.augment_impl(images)


class Strong_Augment(Base_Augment):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)

    def augment_mirror(self, x):
        new_images = x.copy()
        indices = np.arange(len(new_images)).tolist()
        sampled = random.sample(
            indices, int(round(0.5 * len(indices)))
        )  # flip horizontally 50%
        new_images[sampled] = np.fliplr(new_images[sampled])
        return new_images  # random shift

    def augment_shift_mnist(self, x, w):
        y = tf.pad(x, [[0] * 2, [w] * 2, [w] * 2], mode="REFLECT")
        return tf.image.random_crop(y, tf.shape(x))

    def __call__(self, images: np.ndarray):
        return self.augment_shift_mnist(self.augment_mirror(images), 4)


class RandAugment(Base_Augment):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)
        self.rand_augment = Rand_Augment()
        self.input_shape = (32, 32, 3)

    def __call__(self, images):
        print("images:", images.shape)

        return np.array(
            [
                np.array(
                    self.rand_augment(
                        Image.fromarray(np.reshape(img, self.input_shape))
                    )
                )
                for img in images
            ]
        )
=======
version https://git-lfs.github.com/spec/v1
oid sha256:95aa52b51c4d33f4e8385aa86f6bba42b1d2709441ea79f3a36e8ae1bc290c11
size 8436
>>>>>>> 9676c3e (ya toh aar ya toh par)
