# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import torch
import numpy as np
import random
from sklearn.decomposition import PCA

def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]


def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def hflip(vid):
    return vid.flip(dims=(-1,))


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)

def random_resize(vid, size, random_factor, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    r = 1 + random.random() * (random_factor - 1)
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:]) * r
        size = None
    else:
        size = tuple([int(elem * r) for elem in list(size)])
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)

def pad(vid, padding, fill=0, padding_mode="constant"):
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std

def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid
    
def minmax_denormalize(vid, vmin, vmax, scale=2):
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin

def add_noise(data, snr):
    sig_avg_power_db = 10*np.log10(np.mean(data**2))
    noise_avg_power_db = sig_avg_power_db - snr
    noise_avg_power = 10**(noise_avg_power_db/10)
    noise = np.random.normal(0, np.sqrt(noise_avg_power), data.shape)
    noisy_data = data + noise
    
    return noisy_data      

def log_transform(data, k=1, c=0):
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)

def log_transform_tensor(data, k=1, c=0):
    return (torch.log1p(torch.abs(k * data) + c)) * torch.sign(data)

def exp_transform(data, k=1, c=0):
    return (np.expm1(np.abs(data)) - c) * np.sign(data) / k

def tonumpy_denormalize(vid, vmin, vmax, exp=True, k=1, c=0, scale=2):
    if exp:
        vmin = log_transform(vmin, k=k, c=c) 
        vmax = log_transform(vmax, k=k, c=c) 
    vid = minmax_denormalize(vid.cpu().numpy(), vmin, vmax, scale)
    return exp_transform(vid, k=k, c=c) if exp else vid

# Class interface

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return crop(vid, i, j, h, w)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return center_crop(vid, self.size)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

class RandomResize(object):
    def __init__(self, size, random_factor=1.25):
        self.size = size
        self.factor = random_factor

    def __call__(self, vid):
        return random_resize(vid, self.size, self.factor)

class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)

class MinMaxNormalize(object):
    def __init__(self, datamin, datamax, scale=2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale

    def __call__(self, vid):
        return minmax_normalize(vid, self.datamin, self.datamax, self.scale)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return hflip(vid)
        return vid

class Pad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return pad(vid, self.padding, self.fill)
        
class TemporalDownsample(object):
    def __init__(self, rate=1):
        self.rate = rate
        
    def __call__(self, vid):
        return vid[::self.rate]

class AddNoise(object):
    def __init__(self, snr=10):
        self.snr = snr
        
    def __call__(self, vid):
        return add_noise(vid, self.snr)
        
class PCD(object):
    def __init__(self, n_comp=8):
        self.pca = PCA(n_components=n_comp)
    
    def __call__(self, data):
        data= data.reshape((data.shape[0], -1))
        feat_mean = data.mean(axis=0)
        data -= np.tile(feat_mean, (data.shape[0], 1))
        pc = self.pca.fit_transform(data)
        pc = pc.reshape((-1,))
        pc = pc[:, np.newaxis, np.newaxis]
        
        return pc
        
class StackPCD(object):
    def __init__(self, n_comp=(32, 8)):
        
        self.primary_pca = PCA(n_components=n_comp[0])
        self.secondary_pca = PCA(n_components=n_comp[1])
    
    def __call__(self, data):
        
        data = np.transpose(data, (0, 2, 1))
        
        primary_pc = []
        for sample in data:
            feat_mean = sample.mean(axis=0)
            sample -= np.tile(feat_mean, (sample.shape[0], 1))
            primary_pc.append(self.primary_pca.fit_transform(sample))
        primary_pc = np.array(primary_pc)
        
        data = primary_pc.reshape((data.shape[0], -1))
        feat_mean = data.mean(axis=0)
        data -= np.tile(feat_mean, (data.shape[0], 1))
        secondary_pc = self.secondary_pca.fit_transform(data)
        secondary_pc = secondary_pc.reshape((-1,))
        secondary_pc = pc[:, np.newaxis, np.newaxis]
        
        return secondary_pc
    
class LogTransform(object):
    def __init__(self, k=1, c=0):
        self.k = k
        self.c = c

    def __call__(self, data):
        return log_transform(data, k=self.k, c=self.c)
 
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    # def __init__(self, device):
    #     self.device = device
    def __call__(self, sample):
        return torch.from_numpy(sample)
