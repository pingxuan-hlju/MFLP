import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.001, 'learning rate'),
        'dropout': (0.5, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (200, 'maximum number of epochs to train for'),
        'weight-decay': (0.01, 'l2 regularization strength'),
        'optimizer': ('RiemannianAdam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1206, 'seed for training'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (1, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs')
    },
    'model_config': {
        'model': ('GHGNN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]'),
        'dim': (64, 'embedding dimension'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (16, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.5, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (1, 'whether to use hyperbolic attention or not'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
