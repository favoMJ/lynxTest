import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, camvid_colorize_mask


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):
    if output_grey:
        row, col = output.shape
        dst = np.ones((row, col), dtype=np.uint8) * 255
        for i in range(19):
            dst[output == i] = color_list[i]
        output_grey = Image.fromarray(dst)
        output_grey.save(os.path.join(save_path, img_name[:-1] + '.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)
        output_color.save(os.path.join(save_path, img_name[:-1] + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters
