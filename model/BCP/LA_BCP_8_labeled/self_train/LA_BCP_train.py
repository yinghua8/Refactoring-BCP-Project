from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from yaml import parse
from dataloaders.dataset import *

import save_load_net 
from LA_training import LA_train

saveLoad = save_load_net.save_load_net()
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='D:\Master First Semester\Software Design\Final_project\BCP-main\data', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='BCP', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.get_autocast_cpu_dtype() #.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

def log_train_info(path):
    logging.basicConfig(filename = path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/BCP/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('./code/LA_BCP_train.py', self_snapshot_path)

    train_class = LA_train(args, num_classes, patch_size)
    # -- Pre-Training
    log_train_info(pre_snapshot_path)
    train_class.pre_train(pre_snapshot_path)
  
    # -- Self-training
    log_train_info(self_snapshot_path)
    train_class.self_train(pre_snapshot_path, self_snapshot_path, num_classes, patch_size)
