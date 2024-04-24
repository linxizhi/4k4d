import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    eps = cfg.train.eps

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        if key.startswith('all_timestep_pcds'):
            lr=1e-5
        elif key.startswith('geometry_mlp'):
            lr=5e-4
        else:
            lr=cfg.train.lr
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "eps": eps}]

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
