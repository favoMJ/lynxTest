import math
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
import torch


class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            # print(self.base_lrs[0]*warmup_factor)
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            lr = super().get_lr()
        return lr


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_factor=1.0 / 3, warmup_iters=500,
                 eta_min=0, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            # print(self.base_lrs[0]*warmup_factor)
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(
                        math.pi * (self.last_epoch - self.warmup_iters) / (self.T_max - self.warmup_iters))) / 2
                    for base_lr in self.base_lrs]



class WarmupPolyLR(_LRScheduler):#https://blog.csdn.net/sinat_36618660/article/details/99650804
    def __init__(self, optimizer, T_max, cur_iter, warmup_factor=1.0 / 3, warmup_iters=5,
                 eta_min=0, power=0.9):

        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.power = power
        self.T_max, self.eta_min = T_max, eta_min
        self.cur_iter = cur_iter
        super().__init__(optimizer)

    def get_lr(self):#从最初的小学习率开始，每个step增大一点点，直到达到最初设置的比较大的学习率时，采用最初设置的学习率进行训练。
        if self.cur_iter <= self.warmup_iters:
            alpha = self.cur_iter / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            # print(self.base_lrs[0]*warmup_factor)
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    math.pow(1 - (self.cur_iter - self.warmup_iters) / (self.T_max - self.warmup_iters),
                             self.power) for base_lr in self.base_lrs]


def poly_learning_rate(cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    cur_iter = cur_epoch * perEpoch_iter + curEpoch_iter
    max_iter = max_epoch * perEpoch_iter
    lr = baselr * pow((1 - 1.0 * cur_iter / max_iter), 0.9)

    return lr



class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        min_lr_mul: target learning rate = base lr * min_lr_mul
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, min_lr_mul=0.1, after_scheduler=None):
        self.min_lr_mul = min_lr_mul
        if self.min_lr_mul > 1. or self.min_lr_mul < 0.:
            raise ValueError('min_lr_mul should be [0., 1.]')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            else:
                return self.base_lrs
        else:
            return [base_lr * (self.min_lr_mul + (1. - self.min_lr_mul) * (self.last_epoch / float(self.total_epoch))) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio

class WarmupPolyLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            power,
            max_iter,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio



if __name__ == '__main__':
    optim = WarmupPolyLR()
