import cv2
import PIL
import mmcv
import numpy as np
from PIL import Image
import PIL.ImageOps, PIL.ImageDraw, PIL.ImageEnhance
from collections import OrderedDict

FILL_COLOR = (0, 0, 0)


def xyxy_to_xywh(boxes):
    width = boxes[2] - boxes[0]
    height = boxes[3] - boxes[1]
    return [boxes[0], boxes[1], width, height]


def xywh_to_xyxy(boxes):
    x = boxes[0] + boxes[2]
    y = boxes[1] + boxes[3]
    return [boxes[0], boxes[1], x, y]


def draw_bboxes(save_dir, img, bboxes):
    img_copy = img.copy()
    if bboxes:
        for bbox in bboxes:
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
    print('show image shape: ', img_copy.shape)
    cv2.imwrite(save_dir, img_copy)


def TTA_Resize(img, v=(800, 1333), bboxes=None):
    img = img.copy()
    h, w = img.shape[0], img.shape[1]
    m = min(max(v) / max(h, w), min(v) / min(h, w))
    img_aug = cv2.resize(img, (int(w * m + 0.5), int(h * m + 0.5)), cv2.INTER_CUBIC)
    # img_aug = padding_square(img_aug, v)
    bboxes_aug = []
    if bboxes:
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(i * m + 0.5) for i in bbox[:]]
            bboxes_aug.append([x1, y1, x2, y2])
    return img_aug, bboxes_aug, m


def TTA_Resize_mmcv(img, v, bboxes=None):
    aug_img, scale_factor = mmcv.imrescale(img, v, return_scale=True)
    return aug_img, None, scale_factor


def TTA_Resize_re(bboxes, v):
    bboxes_aug = []
    for bbox in bboxes:
        bbox_aug = [int(i / v + 0.5) for i in bbox]
        bboxes_aug.append(bbox_aug)
    return bboxes_aug


def TTA_Flip(img, v, bboxes=None):
    img = img.copy()
    width = img.shape[1]
    height = img.shape[0]
    bboxes_aug = []
    if v == 0:
        img_aug = cv2.flip(img, 0)
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(i) for i in bbox[:]]
                h = y2 - y1
                y1 = height - y2
                y2 = y1 + h
                bboxes_aug.append([x1, y1, x2, y2])
    elif v == 1:
        img_aug = cv2.flip(img, 1)
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(i) for i in bbox[:]]
                w = x2 - x1
                x1 = width - x2
                x2 = x1 + w
                bboxes_aug.append([x1, y1, x2, y2])
    else:
        img_aug = cv2.flip(img, -1)
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(i) for i in bbox[:]]
                w = x2 - x1
                h = y2 - y1
                x1 = width - x2
                y1 = height - y2
                x2 = x1 + w
                y2 = y1 + h
                bboxes_aug.append([x1, y1, x2, y2])
    return img_aug, bboxes_aug, [v, img_aug.shape[1], img_aug.shape[0]]


def TTA_Flip_re(bboxes, v):
    m = v[0]
    width = v[1]
    height = v[2]
    if m == 0:
        bboxes_re = []
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(i) for i in bbox[:]]
                h = y2 - y1
                y1 = height - y2
                y2 = y1 + h
                bboxes_re.append([x1, y1, x2, y2])
    elif m == 1:
        bboxes_re = []
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(i) for i in bbox[:]]
                w = x2 - x1
                x1 = width - x2
                x2 = x1 + w
                bboxes_re.append([x1, y1, x2, y2])
    else:
        bboxes_re = []
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(i) for i in bbox[:]]
                w = x2 - x1
                h = y2 - y1
                x1 = width - x2
                y1 = height - y2
                x2 = x1 + w
                y2 = y1 + h
                bboxes_re.append([x1, y1, x2, y2])
    return bboxes_re


def TTA_Rotate_no_pad(img, v, bboxes=None):
    img = img.copy()
    h, w = img.shape[:2]
    bboxes_aug = []
    if v == 90:
        img_aug = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if bboxes:
            for box in bboxes:
                bboxes_aug.append([h - box[1], box[0], h - box[3], box[2]])
    elif v == 180:
        img_aug = cv2.rotate(img, cv2.ROTATE_180)
        if bboxes:
            for box in bboxes:
                bboxes_aug.append([w - box[0], h - box[1], w - box[2], h - box[3]])
    else:
        img_aug = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if bboxes:
            for box in bboxes:
                bboxes_aug.append([box[1], w - box[0], box[3], w - box[2]])
    return img_aug, bboxes_aug, [v, img_aug.shape[0], img_aug.shape[1]]


def TTA_Rotate_no_pad_re(bboxes, v):
    bboxes_re = []
    if bboxes:
        if v[0] == 90:
            for box in bboxes:
                bboxes_re.append([box[1], v[2] - box[0], box[3], v[2] - box[2]])
        elif v[0] == 180:
            for box in bboxes:
                bboxes_re.append([v[2] - box[0], v[1] - box[1], v[2] - box[2], v[1] - box[3]])
        else:
            for box in bboxes:
                bboxes_re.append([v[1] - box[1], box[0], v[1] - box[3], box[2]])
    return bboxes_re


def TTA_Color(img, v, bboxes=None):  # (0, 1)
    img = Image.fromarray(img.copy())
    img_aug = PIL.ImageEnhance.Color(img).enhance(v)
    return np.array(img_aug), bboxes, v


def TTA_Color_mmcv(img, v, bboxes=None):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.tile(gray_img[..., None], [1, 1, 3])
    beta = 1 - v
    colored_img = cv2.addWeighted(img, v, gray_img, beta, 0)
    if not colored_img.dtype == np.uint8:
        colored_img = np.clip(colored_img, 0, 255)
    return colored_img.astype(img.dtype), bboxes, v


def TTA_Color_re(bboxes, v):
    return bboxes


def TTA_Contrast(img, v, bboxes=None):  # (0, 1)
    img = Image.fromarray(img.copy())
    img_aug = PIL.ImageEnhance.Contrast(img).enhance(v)
    return np.array(img_aug), bboxes, v


def TTA_Contrast_mmcv(img, v, bboxes=None):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.histogram(gray_img, 256, (0, 255))[0]
    mean = round(np.sum(gray_img) / np.sum(hist))
    degenerated = (np.ones_like(img[..., 0]) * mean).astype(img.dtype)
    degenerated = cv2.cvtColor(degenerated, cv2.COLOR_GRAY2BGR)
    contrasted_img = cv2.addWeighted(
        img.astype(np.float32), v, degenerated.astype(np.float32),
        1 - v, 0)
    contrasted_img = np.clip(contrasted_img, 0, 255)
    return contrasted_img.astype(img.dtype), bboxes, v


def TTA_Contrast_re(bboxes, v):
    return bboxes


def TTA_Brightness(img, v, bboxes=None):  # (0, 1)
    img = Image.fromarray(img.copy())
    img_aug = PIL.ImageEnhance.Brightness(img).enhance(v)
    return np.array(img_aug), bboxes, v


def TTA_Brightness_mmcv(img, v, bboxes=None):
    degenerated = np.zeros_like(img)
    brightened_img = cv2.addWeighted(
        img.astype(np.float32), v, degenerated.astype(np.float32),
        1 - v, 0)
    brightened_img = np.clip(brightened_img, 0, 255)
    return brightened_img.astype(img.dtype), bboxes, v


def TTA_Brightness_re(bboxes, v):
    return bboxes


def TTA_Sharpness(img, v, bboxes=None):
    img = Image.fromarray(img.copy())
    img_aug = PIL.ImageEnhance.Sharpness(img).enhance(v)
    return np.array(img_aug), bboxes, v


def TTA_Sharpness_re(bboxes, v):
    return bboxes


def TTA_SHarpness_mmcv(img, v, bboxes=None):
    kernel = np.array([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]) / 13
    degenerated = cv2.filter2D(img, -1, kernel)
    sharpened_img = cv2.addWeighted(
        img.astype(np.float32), v, degenerated.astype(np.float32),
        1 - v, 0)
    sharpened_img = np.clip(sharpened_img, 0, 255)
    return sharpened_img.astype(img.dtype), bboxes, v


def TTA_AutoContrast(img, v, bboxes=None):
    img = Image.fromarray(img.copy())
    cutoff = abs(v)
    img_aug = PIL.ImageOps.autocontrast(img, cutoff)
    return np.array(img_aug), bboxes, v


def TTA_AutoContrast_re(bboxes, v):
    return bboxes


def TTA_AutoContrast_mmcv(img, v, bboxes=None):
    def _auto_contrast_channel(im, c, cutoff):
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, (0, 255))[0]
        # Remove cut-off percent pixels from histo
        histo_sum = np.cumsum(histo)
        cut_low = histo_sum[-1] * cutoff[0] // 100
        cut_high = histo_sum[-1] - histo_sum[-1] * cutoff[1] // 100
        histo_sum = np.clip(histo_sum, cut_low, cut_high) - cut_low
        histo = np.concatenate([[histo_sum[0]], np.diff(histo_sum)], 0)

        # Compute mapping
        low, high = np.nonzero(histo)[0][0], np.nonzero(histo)[0][-1]
        # If all the values have been cut off, return the origin img
        if low >= high:
            return im
        scale = 255.0 / (high - low)
        offset = -low * scale
        lut = np.array(range(256))
        lut = lut * scale + offset
        lut = np.clip(lut, 0, 255)
        return lut[im]

    if isinstance(v, (int, float)):
        cutoff = (v, v)
    else:
        assert isinstance(v, tuple), 'cutoff must be of type int, ' \
                                     f'float or tuple, but got {type(v)} instead.'
    # Auto adjusts contrast for each channel independently and then stacks
    # the result.
    s1 = _auto_contrast_channel(img, 0, cutoff)
    s2 = _auto_contrast_channel(img, 1, cutoff)
    s3 = _auto_contrast_channel(img, 2, cutoff)
    contrasted_img = np.stack([s1, s2, s3], axis=-1)
    return contrasted_img.astype(img.dtype), bboxes, v


def TTA_Equalize(img, v, bboxes=None):
    img = Image.fromarray(img.copy())
    img_aug = PIL.ImageOps.equalize(img)
    return np.array(img_aug), bboxes, v


def TTA_Equalize_mmcv(img, v, bboxes=None):
    def _scale_channel(im, c):
        im = im[:, :, c]
        histo = np.histogram(im, 256, (0, 255))[0]
        nonzero_histo = histo[histo > 0]
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        if not step:
            lut = np.array(range(256))
        else:
            lut = (np.cumsum(histo) + (step // 2)) // step
            lut = np.concatenate([[0], lut[:-1]], 0)
            lut[lut > 255] = 255
        return np.where(np.equal(step, 0), im, lut[im])

    s1 = _scale_channel(img, 0)
    s2 = _scale_channel(img, 1)
    s3 = _scale_channel(img, 2)
    equalized_img = np.stack([s1, s2, s3], axis=-1)
    return equalized_img.astype(img.dtype), bboxes, v


def TTA_Equalize_re(bboxes, v):
    return bboxes


def TTA_Grey(img, v, bboxes=None):
    img = img.copy()
    img_aug = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_aug = cv2.cvtColor(img_aug, cv2.COLOR_GRAY2BGR)
    return img_aug, bboxes, v


def TTA_Grey_re(bboxes, v):
    return bboxes


def TTA_Invert(img, v, bboxes=None):
    img = img.copy()
    img_aug = 255 - img
    return img_aug, bboxes, v


def TTA_Invert_re(bboxes, v):
    return bboxes


def TTA_Posterize(img, v, bboxes=None):
    v = int(8 - abs(v) * 7)
    img = Image.fromarray(img.copy())
    img_aug = PIL.ImageOps.posterize(img, v)
    return np.array(img_aug), bboxes, v


def TTA_Posterize_mmcv(img, v, bboxes=None):
    shift = 8 - int(abs(8 * v))
    img = np.left_shift(np.right_shift(img, shift), shift)
    return img, bboxes, v


def TTA_Posterize_re(bboxes, v):
    return bboxes


def TTA_Solarize(img, v, bboxes=None):
    v = int((1 - abs(v)) * 255)
    img = Image.fromarray(img.copy())
    img_aug = PIL.ImageOps.solarize(img, v)
    return np.array(img_aug), bboxes, v


def TTA_Solarize_mmcv(img, v, bboxes=None):
    v = int((1 - abs(v)) * 255)
    img = np.where(img < v, img, 255 - img)
    return img, bboxes, v


def TTA_Solarize_re(bboxes, v):
    return bboxes


def TTA_HSV(img, v, bboxes=None):
    img = img.copy()
    img_aug = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_aug, bboxes, v


def TTA_HSV_re(bboxes, v):
    return bboxes


def TTA_PepperNoise(img, v, bboxes=None):
    img = img.copy()
    per = abs(v) / 2
    pix = img.shape[0] * img.shape[1]
    num = int(pix * per * 0.5)
    coords = [np.random.randint(0, i - 1, num) for i in img.shape[:2]]
    img[coords[0], coords[1], :] = [255, 255, 255]
    coords = [np.random.randint(0, i - 1, num) for i in img.shape[:2]]
    img[coords[0], coords[1], :] = [0, 0, 0]
    return img, bboxes, v


def TTA_PepperNoise_re(bboxes, v):
    return bboxes


def TTA_GaussNoise(img, v, bboxes=None):
    img = img.copy()
    mean = 0
    sigma = abs(v) * 100
    gauss = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2]))
    noisy_img = img + gauss
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return np.uint8(noisy_img), bboxes, v


def TTA_GaussNoise_re(bboxes, v):
    return bboxes


def printf(x):
    print(x)
    exit(0)


def padding_square(img, out_size):
    h, w = img.shape[:2]
    ret = cv2.copyMakeBorder(img, 0, out_size[0] - h, 0, out_size[1] - w, cv2.BORDER_CONSTANT, value=FILL_COLOR)
    return ret


def topleftxywh_to_xyxy(boxes):
    """
    args:
        boxes:list of topleft_x,topleft_y,width,height,
    return:
        boxes:list of x,y,x,y,corresponding to top left and bottom right
    """
    x_top_left = boxes[0]
    y_top_left = boxes[1]
    x_bottom_right = boxes[0] + boxes[2]
    y_bottom_right = boxes[1] + boxes[3]
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]


def TTA_Aug_List():
    return [TTA_Resize, TTA_Flip, TTA_Rotate_no_pad, TTA_Color, TTA_Contrast, TTA_Brightness, TTA_Sharpness,
            TTA_AutoContrast, TTA_Equalize, TTA_Grey, TTA_Invert, TTA_Posterize, TTA_Solarize, TTA_HSV, TTA_PepperNoise,
            TTA_GaussNoise]


def TTA_Aug_Space(resolution=(640, 640), size_divisor=32):
    '''
        0 TTA_Resize
        1 TTA_Flip
        2 TTA_Rotate_no_pad
        3 TTA_Color
        4 TTA_Contrast
        5 TTA_Brightness
        6 TTA_Sharpness
        7 TTA_AutoContrast
        8 TTA_Equalize
        9 TTA_Grey
        10 TTA_Invert
        11 TTA_Posterize
        12 TTA_Solarize
        13 TTA_HSV
        14 TTA_PepperNoise
        15 TTA_GaussNoise
    '''
    default_aug_space = [
        (1, 0), (1, 1), (1, -1),
        (2, 90), (2, 180), (2, 270),
        (3, 0.2), (3, 0.3), (3, 0.4), (3, 0.5), (3, 0.6), (3, 0.7), (3, 0.8),
        (4, 0.2), (4, 0.3), (4, 0.4), (4, 0.5), (4, 0.6), (4, 0.7), (4, 0.8),
        (5, 0.2), (5, 0.3), (5, 0.4), (5, 0.5), (5, 0.6), (5, 0.7), (5, 0.8),
        (6, 0.2), (6, 0.3), (6, 0.4), (6, 0.5), (6, 0.6), (6, 0.7), (6, 0.8),
        (7, 0.2), (7, 0.3), (7, 0.4), (7, 0.5), (7, 0.6), (7, 0.7), (7, 0.8),
        (8, 1),
        (9, 1),
        (10, 1),
        (11, 0.2), (11, 0.3), (11, 0.4), (11, 0.5), (11, 0.6), (11, 0.7), (11, 0.8),
        (12, 0.2), (12, 0.3), (12, 0.4), (12, 0.5), (12, 0.6), (12, 0.7), (12, 0.8),
        (13, 1),
        (14, 0.2), (14, 0.3), (14, 0.4), (14, 0.5), (14, 0.6), (14, 0.7), (14, 0.8),
        (15, 0.2), (15, 0.3), (15, 0.4), (15, 0.5), (15, 0.6), (15, 0.7), (15, 0.8)
    ]
    forward_resolution_space = [(0, (resolution[0] + (i * size_divisor), resolution[1] + (i * size_divisor))) for i in range(8)]
    backward_resolution_space = [(0, (resolution[0] - (i * size_divisor), resolution[1] - (i * size_divisor))) for i in range(8)]
    forward_resolution_space.extend(backward_resolution_space)
    unique_resolution_space = list(OrderedDict.fromkeys(forward_resolution_space))
    unique_resolution_space.extend(default_aug_space)
    return unique_resolution_space


def Test_Aug_Space():
    return [(0, (672, 672)),
            (1, -1),
            (2, 90),
            (3, 0.4),
            (4, 0.2),
            (5, 0.2),
            (6, 0.2),
            (7, 0.2),
            (8, 1),
            (9, 1),
            ]
