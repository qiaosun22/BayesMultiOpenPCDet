import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt, eval_single_ckpt
from noise import add_noise_to_weights
import numba
import logging
import time

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model, model_save
from eval_utils import eval_utils
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():

    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        print('Using GPU:' + str(np.argmax(memory_gpu)))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
        os.system('rm tmp')
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus


    save_path = './results/pointpillar/'+time.strftime('%m%d-%H%M',time.localtime(time.time()))+'/'

    # output_dir = result_dir / 'final_result' / 'data'
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'bayes' / args.extra_tag


    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True) 
    

    logger = common_utils.create_logger(save_path+'log.txt', rank=cfg.LOCAL_RANK)

    file = open(save_path+'3d.txt','w')
    file.close()

    file = open(save_path+'bev.txt','w')
    file.close()

    file = open(save_path+'aos.txt','w')
    file.close()

    file = open(save_path+'image.txt','w')
    file.close()

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    p1 = p2 = 0

    model = build_network(model_cfg=cfg.MODEL, 
                        num_class=len(cfg.CLASS_NAMES), 
                        p1=1, 
                        p2=1, 
                        dataset=test_set)

    # ckpt_pth = save_path+'bayes_model-{}-{}'.format(p1,p2)
    # ckpt_name = ckpt_pth+'.pth'
    # model_save(model, ckpt_pth, optimizer, args.epochs, args.epochs)

    logger.info('**********************End training**********************')

    # if dist_test: 
    #     model = model.module

    logger.info('----------------load ckpt------'+args.ckpt[-24:])

    
    # if acc > best_accu:
    logger.info('----------------Noise Experiment----------------')

    N = 10
    S = np.linspace(0.4, 0.6, 5)
    with torch.no_grad():
        for s in S:
            for n in range(N):
                model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
                model.cuda()
                add_noise_to_weights(0, s, model)
                print('sigma:{}, evaluate-{}'.format(s, n))
                eval_utils.eval_simple(
                                        cfg, model, test_loader, logger, save_path, dist_test=dist_test,
                                        save_to_file=args.save_to_file, result_dir=output_dir
                                            )

    logger.info('**********************End evaluation**********************')
    



if __name__ == '__main__':
    main()
