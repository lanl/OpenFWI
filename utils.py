# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import errno
import os
import itertools
from torchvision.models import vgg16
import numpy as np

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if isinstance(iterable, list):
            length = max(len(x) for x in iterable)
            iterable = [x if len(x) == length else itertools.cycle(x) for x in iterable]
            iterable = zip(*iterable)
        else:
            length = len(iterable)
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(length))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj # <-- yield the batch in for loop
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (length - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, length, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, length, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


# Legacy code
class ContentLoss(nn.Module):
    def __init__(self, args):
        super(ContentLoss, self).__init__()
        names = ['l1', 'l2']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, model, input, target):
        pred = model(input)
        loss_l1 = self.l1loss(target, pred)
        loss_l2 = self.l2loss(target, pred)
        loss = loss_l1 * self.lambda_l1 + loss_l2 * self.lambda_l2
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }


# Legacy code
class IdenticalLoss(nn.Module):
    def __init__(self, args):
        super(IdenticalLoss, self).__init__()
        names = ['id1s', 'id2s']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, model_s2v, model_v2s, input):
        mid = model_s2v(input)
        pred = model_v2s(mid)
        cal_loss = lambda x, y: (self.l1loss(x, y), self.l2loss(x, y))
        loss_id1s, loss_id2s = cal_loss(input, pred)
        loss = loss_id1s * self.lambda_id1s + loss_id2s * self.lambda_id2s
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }

# Implemented according to H-PGNN, not useful
class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()
    
    def forward(self, pred, gt):
        return torch.mean(((pred - gt) / (torch.amax(gt, (-2, -1), keepdim=True) + 1e-5)) ** 2)


class CycleLoss(nn.Module):
    def __init__(self, args):
        super(CycleLoss, self).__init__()
        names = ['g1v', 'g2v', 'g1s', 'g2s', 'c1v', 'c2v', 'c1s', 'c2s']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
    
    def forward(self, data, label, pred_s=None, pred_v=None, recon_s=None, recon_v=None):
        cal_loss = lambda x, y: (self.l1loss(x, y), self.l2loss(x, y))
        loss_g1v, loss_g2v, loss_g1s, loss_g2s = [0] * 4
        if pred_v is not None:
            loss_g1v, loss_g2v = cal_loss(pred_v, label) 
        if pred_s is not None:
            loss_g1s, loss_g2s = cal_loss(pred_s, data)

        loss_c1v, loss_c2v, loss_c1s , loss_c2s = [0] * 4
        if recon_v is not None:
            loss_c1v, loss_c2v = cal_loss(recon_v, label)
        if recon_s is not None:
            loss_c1s, loss_c2s = cal_loss(recon_s, data)

        loss = loss_g1v * self.lambda_g1v + loss_g2v * self.lambda_g2v + \
            loss_g1s * self.lambda_g1s + loss_g2s * self.lambda_g2s + \
            loss_c1v * self.lambda_c1v + loss_c2v * self.lambda_c2v + \
            loss_c1s * self.lambda_c1s + loss_c2s * self.lambda_c2s
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }


# Legacy code
class _CycleLoss(nn.Module):
    def __init__(self, args):
        super(_CycleLoss, self).__init__()
        names = ['g1v', 'g2v', 'g1s', 'g2s', 'c1v', 'c2v', 'c1s', 'c2s']
        self.loss_names = ['loss_' + n for n in names]
        for key in ['lambda_' + n for n in names]:
            setattr(self, key, getattr(args, key))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
    
    def forward(self, data, label, pred_s=None, pred_v=None, recon_s=None, recon_v=None):
        cal_loss = lambda x, y: (self.l1loss(x, y), self.l2loss(x, y))
        loss_g1v, loss_g2v, loss_g1s, loss_g2s = [0] * 4
        if pred_v is not None and (self.lambda_g1v != 0 or self.lambda_g2v != 0):
            loss_g1v, loss_g2v = cal_loss(pred_v, label) 
        if pred_s is not None and (self.lambda_g1s != 0 or self.lambda_g2s != 0):
            loss_g1s, loss_g2s = cal_loss(pred_s, data)

        loss_c1v, loss_c2v, loss_c1s , loss_c2s = [0] * 4
        if recon_v is not None and (self.lambda_c1v != 0 or self.lambda_c2v != 0):
            loss_c1v, loss_c2v = cal_loss(recon_v, label)
        if recon_s is not None and (self.lambda_c1s != 0 or self.lambda_c2s != 0):
            loss_c1s, loss_c2s = cal_loss(recon_s, data)

        loss = loss_g1v * self.lambda_g1v + loss_g2v * self.lambda_g2v + \
            loss_g1s * self.lambda_g1s + loss_g2s * self.lambda_g2s + \
            loss_c1v * self.lambda_c1v + loss_c2v * self.lambda_c2v + \
            loss_c1s * self.lambda_c1s + loss_c2s * self.lambda_c2s
        scope = locals()
        return loss, { k: eval(k, scope) for k in self.loss_names }

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and args.world_size > 1:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
    
    
class Wasserstein_GP(nn.Module):
    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

# Modified from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49     
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg16(pretrained=True).features[:4].eval()) # relu1_2
        blocks.append(vgg16(pretrained=True).features[4:9].eval()) # relu2_2
        blocks.append(vgg16(pretrained=True).features[9:16].eval()) # relu3_3
        blocks.append(vgg16(pretrained=True).features[16:23].eval()) # relu4_3
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, input, target, rescale=True, feature_layers=[1]):
        input = input.view(-1, 1, input.shape[-2], input.shape[-1]).repeat(1, 3, 1, 1)
        target = target.view(-1, 1, target.shape[-2], target.shape[-1]).repeat(1, 3, 1, 1)
        if rescale: # from [-1, 1] to [0, 1]
            input = input / 2 + 0.5
            target = target / 2 + 0.5
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss_l1, loss_l2 = 0.0, 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss_l1 += self.l1loss(x, y)
                loss_l2 += self.l2loss(x, y)
        return loss_l1, loss_l2


def cal_psnr(gt, data, max_value):
    mse = np.mean((gt - data) ** 2)
    if (mse == 0):
       return 100
    return 20 * np.log10(max_value / np.sqrt(mse))
