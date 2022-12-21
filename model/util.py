from fid_score import calculate_fid_given_paths

import torch
import torch.nn.functional as F

import torchvision.utils as vutils
from torchvision import transforms

import os


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = F.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)


def save_image_list(model_name, dataset, real, best, path):
    if real:
        base_path = './../img/real/{}_{}'.format(model_name, path)
    else:
        base_path = './../img/fake/{}_{}'.format(model_name, path)

    if best:
        base_path = base_path+'_best'

    #dataset_path = []
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    #denormalize = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


    for i in range(len(dataset)):
        save_path = f'{base_path}/image_{i}.png'
        #dataset_path.append(save_path)
        vutils.save_image((dataset[i]+1)/2, save_path)  # denormalize and save image

    return base_path


def get_fid(real_path, fake_path, batch_size, cuda=True, dims=2048):
    return calculate_fid_given_paths([real_path, fake_path],
                                         batch_size,
                                         cuda,
                                         dims)
