import argparse
import datetime
import glob
from pathlib import Path

import numba
import logging
import os

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
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from hardware_noise.new_weight_mapping import weights_mapping as add_noise_to_weights





def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml', \
                        help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=32, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='checkpoint_epoch_80.pth', \
                        help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=29051, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
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



if __name__ == '__main__':
    print(datetime.datetime.now())
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_train = False  # True
        total_gpus = 1
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        print('Using GPU:' + str(np.argmax(memory_gpu)))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
        os.system('rm tmp')
    else:
        args.is_master = args.local_rank == 0
        args.device = torch.cuda.device(args.local_rank)
        torch.cuda.manual_seed_all(666)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5678'
        torch.cuda.set_device(1)
        device = torch.device('cuda', cfg.LOCAL_RANK)
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True


    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger('./baseline/pointpillar/log.txt', rank=cfg.LOCAL_RANK)

    file = open('./baseline/pointpillar/result.txt','w')
    file.write('results\n')
    file.close()


    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))


    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()


    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)





    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs

    if dist_train: 
        model = model.module


    logger.info('----------------Noise Experiment----------------')

    save_path = './save_path/'
    f = open(save_path+'3d.txt', "a+")
    f.write(str(datetime.datetime.now())+'\n')
    f.close()

    f = open(save_path+'3d.txt', "a+")
    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('s', 'n', 'mthd',
                                        'Car/easy_R40',
                                        'Car/moderate_R40',
                                        'Car/hard_R40',
                                        'Pedestrian/easy_R40',
                                        'Pedestrian/moderate_R40',
                                        'Pedestrian/hard_R40',
                                        'Cyclist/easy_R40',
                                        'Cyclist/moderate_R40',
                                        'Cyclist/hard_R40',
                                        'easy_R40',
                                        'moderate_R40',
                                        'hard_R40',
                                        'avg', 'time'
                                        ))
    f.close()




    hw_data_files = os.listdir('./hardware_noise/hardware_data/')

    N = 10
    # S = np.linspace(1e-31, 1.0, 21)#[0.3, 2.0]#
    file2ap_dict = {}
    sigma = 0.01
    for f_name in sorted(hw_data_files)[-5:]:
        if f_name.endswith('xlsx'):
            file2ap_dict[f_name] = {}
            print(f_name)
            for n in range(N):
                model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_train)
                # add_noise_to_weights(0, s, model)
                model = add_noise_to_weights('./hardware_noise/hardware_data/'+f_name, 0, sigma, model, 'cuda')
                print('sigma:{}, evaluate-{}'.format(f_name, n))
                ret_dict = eval_utils.eval_simple(f_name, n, 
                                                cfg, model, test_loader, logger, save_path=None, dist_test=dist_train,
                                                save_to_file=args.save_to_file, result_dir=eval_output_dir
                                            )
                file2ap_dict[f_name][n] = ret_dict

    logger.info('**********************End evaluation %s/%s(%s)**********************')