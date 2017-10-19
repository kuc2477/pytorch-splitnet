#! /usr/bin/env python3
from argparse import ArgumentParser
import torch
from data import TRAIN_DATASETS, TEST_DATASETS, DATASET_CONFIGS
from model import WideResNet
from train import train
import utils


parser = ArgumentParser('WRN torch implementation')
parser.add_argument(
    '--dataset', default='cifar100', type=str,
    choices=list(TRAIN_DATASETS.keys())
)
parser.add_argument('--total-block-number', type=int, default=6)
parser.add_argument('--widen-factor', type=int, default=8)
parser.add_argument(
    '--baseline-strides', type=int, default=[1, 1, 2, 2], nargs='+'
)
parser.add_argument(
    '--baseline-channels', type=int, default=[16, 16, 32, 64], nargs='+'
)
parser.add_argument('--split-sizes', type=int, default=[2, 2, 2], nargs='+')
parser.add_argument('--gamma1', type=float, default=1.)
parser.add_argument('--gamma2', type=float, default=1.)
parser.add_argument('--gamma3', type=float, default=10.)
parser.add_argument('--weight-decay', type=float, default=1e-04)
parser.add_argument('--dropout-prob', type=float, default=.5)
parser.add_argument('--lr', type=float, default=3e-04)
parser.add_argument('--lr-decay', type=float, default=.1)
parser.add_argument('--lr-decay-epochs', type=int, default=[10, 30, 50],
                    nargs='+')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-size', type=int, default=1000)
parser.add_argument('--eval-log-interval', type=int, default=50)
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--weight-log-interval', type=int, default=500)
parser.add_argument('--checkpoint-interval', type=int, default=500)
parser.add_argument('--model-dir', type=str, default='models')
resume_command = parser.add_mutually_exclusive_group()
resume_command.add_argument('--resume-best', action='store_true')
resume_command.add_argument('--resume-latest', action='store_true')
parser.add_argument('--best', action='store_true')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--test', action='store_true', dest='test')
main_command.add_argument('--train', action='store_false', dest='test')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    train_dataset = TRAIN_DATASETS[args.dataset]
    test_dataset = TEST_DATASETS[args.dataset]
    dataset_config = DATASET_CONFIGS[args.dataset]

    # instantiate the model instance.
    wrn = WideResNet(
        args.dataset,
        dataset_config['size'],
        dataset_config['channels'],
        dataset_config['classes'],
        total_block_number=args.total_block_number,
        widen_factor=args.widen_factor,
        dropout_prob=args.dropout_prob,
        baseline_strides=args.baseline_strides,
        baseline_channels=args.baseline_channels,
        split_sizes=args.split_sizes,
    )

    # initialize the weights.
    utils.xavier_initialize(wrn)

    # prepare cuda if needed.
    if cuda:
        wrn.cuda()

    # run the given command.
    if args.test:
        utils.load_checkpoint(wrn, args.model_dir, best=True)
        utils.validate(
            wrn, test_dataset, test_size=args.test_size,
            cuda=cuda, verbose=True
        )
    else:
        train(
            wrn, train_dataset, test_dataset=test_dataset,
            model_dir=args.model_dir,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_decay_epochs=args.lr_decay_epochs,
            weight_decay=args.weight_decay,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
            gamma3=args.gamma3,
            batch_size=args.batch_size,
            test_size=args.test_size,
            epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            eval_log_interval=args.eval_log_interval,
            loss_log_interval=args.loss_log_interval,
            weight_log_interval=args.weight_log_interval,
            resume_best=args.resume_best,
            resume_latest=args.resume_latest,
            cuda=cuda,
        )
