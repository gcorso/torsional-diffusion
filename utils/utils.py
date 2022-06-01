import os

import torch
import yaml, time
from collections import defaultdict
from diffusion.score_model import TensorProductScoreModel


def get_model(args):
    return TensorProductScoreModel(in_node_features=args.in_node_features, in_edge_features=args.in_edge_features,
                                   ns=args.ns, nv=args.nv, sigma_embed_dim=args.sigma_embed_dim,
                                   sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                   num_conv_layers=args.num_conv_layers,
                                   max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim,
                                   scale_by_sigma=args.scale_by_sigma,
                                   use_second_order_repr=args.use_second_order_repr,
                                   residual=not args.no_residual, batch_norm=not args.no_batch_norm)


def get_optimizer_and_scheduler(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


class TimeProfiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.starts = {}
        self.curr = None

    def start(self, tag):
        self.starts[tag] = time.time()

    def end(self, tag):
        self.times[tag] += time.time() - self.starts[tag]
        del self.starts[tag]
