import torch.optim as optim
from .warmup import WarmupLR
import torch.optim.lr_scheduler as lr_scheduler


def build_optimizer(self):
    if self.hparams.optimizer.lower() == 'adam':
        optimizer = optim.Adam(params=self.parameters(), lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay, betas=self.hparams.betas, eps=1e-8)
    elif self.hparams.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(params=self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay, betas=self.hparams.betas, eps=1e-8)
    elif self.hparams.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(params=self.parameters(), lr=self.hparams.learning_rate,
                              momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
    elif self.hparams.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(params=self.parameters(), lr=self.hparams.learning_rate, alpha=self.hparams.alpha,
                                  weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, centered=self.hparams.centered)
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(self):
    if self.hparams.scheduler.lower() == 'step':
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
    elif self.hparams.scheduler.lower() == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
    elif self.hparams.scheduler.lower() == 'exp':
        scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.hparams.gamma)
    elif self.hparams.scheduler.lower() == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
    else:
        raise NotImplementedError(f'Not support type: {self.hparams.scheduler}, '
                                  f'you can design the scheduler by lr_scheduler.LambdaLR.')
    scheduler = WarmupLR(scheduler, init_lr=self.hparams.init_lr, num_warmup=self.hparams.warmup_epochs,
                         warmup_strategy=self.hparams.warmup_strategy)

    return scheduler
