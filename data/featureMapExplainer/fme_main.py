# Created by jing at 26.06.23
import os
import json
import argparse
import datetime
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader

import config
from data.featureMapExplainer.engine import train_engine, dataset
from data.featureMapExplainer.patterns import p_color_net, p_color
def load_args_from_file(args_file_path, given_args):
    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)

        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            # if key not in ['conflict_th', 'sc_th','nc_th']:  # Do not overwrite these keys
            setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))
    return None


def paser(args_path):
    """
    Parese command line arguments

    Args:
    opt_args: Optional args for testing the function. By default, sys.argv is used

    Returns:
        args: Dictionary of args.

    Raises:
        ValueError: Raises an exception if some arg values are invalid.
    """
    # Construct the parser
    parser = argparse.ArgumentParser()
    # Mode selection
    parser.add_argument('--device', type=str, default="cpu", help='choose the training device, cpu or cuda:0')
    parser.add_argument('--net_type', type=str, help='choose the net for training')
    parser.add_argument('--num-channels', type=int, help='choose the number of channels in the model')
    parser.add_argument('--exp', '--e', help='Experiment name')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')

    ########### General Dataset arguments ##########
    parser.add_argument('--dataset', type=str, default='', help='Dataset Name.')
    parser.add_argument('--batch_size', '-b', default=4, type=int, help='Mini-batch size (default: 4)')
    parser.add_argument('--train-on', default='full', type=str, help='The number of images to train on from the data.')

    ########### Training arguments ###########
    parser.add_argument('--epochs', default=20, type=int,
                        help='Total number of epochs to run (default: 30)')
    parser.add_argument('--optimizer', '-o', default='adam')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='Initial learning rate (default 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum.')
    parser.add_argument('--lr-scheduler', default="100,1000", type=str, help='lr schedular.')
    parser.add_argument('--lr_decay_factor', default=0.5, type=float, help='lr decay factor.')
    parser.add_argument('--loss', '-l', default='l1')
    parser.add_argument('--loss-type', default='l2')
    parser.add_argument('--init-net', type=str, default=None)

    ########### Logging ###########
    parser.add_argument('--print-freq', default=100, type=int,
                        help='Printing evaluation criterion frequency (default: 10)')
    # Parse the arguments
    args = parser.parse_args()

    # Path to the workspace directory
    arg_file = args_path / f"{args.exp}.json"
    load_args_from_file(arg_file, args)

    return args


def load_network(args):
    if args.exp == "p_color":
        network = p_color_net.CNN(3, len(p_color.label_map))
    else:
        raise ValueError
    return network


def init():
    # init args
    args = paser(config.exp_path)
    args.data_path = config.data_path / args.exp
    args.start_date = datetime.datetime.today().date()
    args.start_time = datetime.datetime.now().strftime("%H_%M_%S")
    args.eval_loss_best = 1e+10
    args.is_best = False
    args.output_path = args.data_path / "output"
    args.analysis_path = args.data_path / "analysis"
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists(args.analysis_path):
        os.mkdir(args.analysis_path)
    args.FEATURE_MAP_NUM = 2
    args.CV_COLOR = cv.COLORMAP_TURBO
    # args.CV_COLOR = cv.COLORMAP_DEEPGREEN

    # init network
    network = load_network(args)

    # init model
    if args.resume == None:
        args.start_epoch = 0
        args.model = network.to(0 if torch.cuda.is_available() else "cpu")
        args.parameters = filter(lambda p: p.requires_grad, args.model.parameters())
        args.optimizer = Adam(args.parameters, lr=args.lr, weight_decay=0, amsgrad=True)
        args.criterion = torch.nn.MSELoss(reduction='mean')
        args.milestones = [int(x) for x in args.lr_scheduler.split(",")]
        args.lr_decayer = lr_scheduler.MultiStepLR(args.optimizer, milestones=args.milestones,
                                                   gamma=args.lr_decay_factor)
    else:
        raise NotImplemented

    # init dataset
    args.train_loader, args.test_loader = dataset.create_dataloader(args)

    # logs
    print(f'- Training Device: {args.device}, Date: {datetime.datetime.today().date()}\n')

    return args


if __name__ == '__main__':
    args =  init()
    train_engine.main(args)
