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
from pcdet.models_bayes import build_network, model_fn_decorator
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
    parser.add_argument('--cfg_file', type=str, default='./cfgs/kitti_models/pointpillar_bayes.yaml', \
                        help='specify the config for training')
    # sunqiao/OpenPCDet/tools/cfgs/kitti_models/pointpillar_bayes.yaml
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=32, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='../output/cfgs/kitti_models/bayes/default/ckpt/checkpoint_epoch_33.pth', help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=True, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
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


def opt_function(p1, p2):
    
    print("=============")
    print(p1, p2)
    print("=============")
    
    global best_accu

    # p1 = round(p1, 2)
    # p2 = round(p2, 2)

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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), 
                        p1=p1, 
                        p2=p2, 
                        dataset=train_set)
    # print(model.state_dict())
    # print("???????????")
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is True:
        model.load_params_from_file(filename=args.ckpt, to_cpu=dist, logger=logger)

    # if args.ckpt is not None:
    #     it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
    #     last_epoch = start_epoch + 1
    # else:
    #     ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    #     if len(ckpt_list) > 0:
    #         ckpt_list.sort(key=os.path.getmtime)
    #         it, start_epoch = model.load_params_with_optimizer(
    #             ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
    #         )
    #         last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

#     # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    test_set, test_loader, sampler = build_dataloader(
                                    dataset_cfg=cfg.DATA_CONFIG,
                                    class_names=cfg.CLASS_NAMES,
                                    batch_size=args.batch_size,
                                    dist=dist_train, workers=args.workers, logger=logger, training=False
                                )
    
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), p1=0.11, p2=0.11, dataset=test_set)
    if dist_train: 
        model = model.module

    if args.pretrained_model is True:
        model.load_params_from_file(filename=args.ckpt, to_cpu=dist, logger=logger)
        
    # model.load_params_from_file(filename='./checkpoint_epoch_80.pth', logger=logger, to_cpu=dist_train)
    model.cuda()

    ckpt_pth = save_path+'bayes_model-{}-{}'.format(p1,p2)
    ckpt_name = ckpt_pth+'.pth'

    if cfg.LOCAL_RANK == 0:
        model_save(model, ckpt_pth, optimizer, args.epochs, args.epochs)

    logger.info('**********************End training**********************')

    time.sleep(30)



    # if dist_train: 
    #     model = model.module

    sigma = 0.1
    f = open(save_path+'result.txt', "a+")
    # f.write('----------------Noise-{}-evaluate----------------'.format(sigma))
    f.write('----------------{}-{}----------------\n'.format(p1,p2))
    f.close()

    logger.info('----------------Noise-{}-evaluate----------------'.format(sigma))
    # model.load_params_from_file(filename=ckpt_name, logger=logger, to_cpu=dist_train)
    # model.cuda()
    add_noise_to_weights(0, sigma, model)
    global n
    n += 1
    
    acc1 = eval_utils.eval_simple(sigma, n, cfg, model, test_loader, logger, save_path, dist_test=dist_train, save_to_file=args.save_to_file, result_dir=eval_output_dir)
    print("----------")
    print(acc1)
    print("----------")

#     model.load_params_from_file(filename=ckpt_name, logger=logger, to_cpu=dist_train)
#     model.cuda()
#     add_noise_to_weights(0, sigma, model)
#     acc2 = eval_utils.eval_simple('1', '1', cfg, model, test_loader, logger, save_path, dist_test=dist_train,
#                                         save_to_file=args.save_to_file, result_dir=eval_output_dir)


#     model.load_params_from_file(filename=ckpt_name, logger=logger, to_cpu=dist_train)
#     model.cuda()
#     add_noise_to_weights(0, sigma, model)
#     acc3 = eval_utils.eval_simple('1', '1', cfg, model, test_loader, logger, save_path, dist_test=dist_train,
#                                         save_to_file=args.save_to_file, result_dir=eval_output_dir)

#     f = open(save_path+'result.txt', "a+")
#     f.write('----------------Noise-{}-evaluate----------------\n'.format(sigma))
#     f.close()

    # acc = (ret_dict['Car_3d/easy_R40'] + \
    #     ret_dict['Pedestrian_3d/easy_R40'] + \
    #     ret_dict['Cyclist_3d/easy_R40'] + \
    #     ret_dict['Car_3d/moderate_R40'] + \
    #     ret_dict['Pedestrian_3d/moderate_R40'] + \
    #     ret_dict['Cyclist_3d/moderate_R40'] + \
    #     ret_dict['Car_3d/hard_R40'] + \
    #     ret_dict['Pedestrian_3d/hard_R40'] + \
    #     ret_dict['Cyclist_3d/hard_R40'])/9

    # print(acc)

    # acc = 0.3

    # if acc > best_accu:
#     logger.info('----------------Noise Experiment----------------')

#     N = 5
#     S = np.linspace(0., 0.5, 6)

#     for s in S:
#         for n in range(N):
#             model.load_params_from_file(filename=ckpt_name, logger=logger, to_cpu=dist_train)
#             model.cuda()
#             add_noise_to_weights(0, s, model)
#             print('sigma:{}, evaluate-{}'.format(s, n))
#             eval_utils.eval_simple(
#                                     cfg, model, test_loader, logger, save_path, dist_test=dist_train,
#                                     save_to_file=args.save_to_file, result_dir=eval_output_dir
#                                         )

    logger.info('**********************End evaluation**********************')

        # best_accu = acc

    return acc1#+acc2+acc3


# def main():

#     global best_accu
#     best_accu = 0

#     args, cfg = parse_config()
#     if args.launcher == 'none':
#         dist_train = False
#         total_gpus = 1
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#         memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#         print('Using GPU:' + str(np.argmax(memory_gpu)))
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
#         os.system('rm tmp')
#     else:
#         total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
#             args.tcp_port, args.local_rank, backend='nccl'
#         )
#         dist_train = True

#     if args.batch_size is None:
#         args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
#     else:
#         assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
#         args.batch_size = args.batch_size // total_gpus

#     args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

#     if args.fix_random_seed:
#         common_utils.set_random_seed(666)

#     output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'bayes' / args.extra_tag
#     ckpt_dir = output_dir / 'ckpt'
#     output_dir.mkdir(parents=True, exist_ok=True)
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     save_path = './bayes/pointpillar/'+time.strftime('%m%d-%H%M',time.localtime(time.time()))+'/'

#     if not os.path.exists(save_path):
#         os.makedirs(save_path, exist_ok=True) 

#     logger = common_utils.create_logger(save_path+'log.txt', rank=cfg.LOCAL_RANK)

#     file = open(save_path+'result.txt','w')
#     file.write('results\n')
#     file.close()

#     # head = ''
#     # logging.basicConfig(filename='./baseline/pointpillar/log.txt',
#     #                     format=head)
#     # logger_result = logging.getLogger()
#     # logger_result.setLevel(logging.INFO)
#     # console = logging.StreamHandler()
#     # logging.getLogger('').addHandler(console)

#     # log to file
#     logger.info('**********************Start logging**********************')
#     gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
#     logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

#     if dist_train:
#         logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
#     for key, val in vars(args).items():
#         logger.info('{:16} {}'.format(key, val))
#     log_config_to_file(cfg, logger=logger)
#     if cfg.LOCAL_RANK == 0:
#         os.system('cp %s %s' % (args.cfg_file, output_dir))

#     tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

#     # -----------------------create dataloader & network & optimizer---------------------------
#     # train_set, train_loader, train_sampler = build_dataloader(
#     #     dataset_cfg=cfg.DATA_CONFIG,
#     #     class_names=cfg.CLASS_NAMES,
#     #     batch_size=args.batch_size,
#     #     dist=dist_train, workers=args.workers,
#     #     logger=logger,
#     #     training=True,
#     #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
#     #     total_epochs=args.epochs
#     # )


#     # test_set, test_loader, sampler = build_dataloader(
#     #     dataset_cfg=cfg.DATA_CONFIG,
#     #     class_names=cfg.CLASS_NAMES,
#     #     batch_size=args.batch_size,
#     #     dist=dist_train, workers=args.workers, logger=logger, training=False
#     # )
#     eval_output_dir = output_dir / 'eval' / 'eval_with_train'
#     eval_output_dir.mkdir(parents=True, exist_ok=True)
#     args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs


#     logger.info('----------------Bayes Optimization----------------')
#     # Bounded region of parameter space
#     pbounds = {'p1': (0.15, 0.8), 'p2': (0.15, 0.8)}

#     optimizer = BayesianOptimization(
#         f=opt_function,
#         pbounds=pbounds,
#         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#         random_state=1,
#     )
#     optimizer.probe(
#         params={'p1': 0.15, 'p2': 0.15},
#         lazy=True,
#     )

#     logger_bayes = JSONLogger(path=save_path+"logs2.json")
#     optimizer.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)

#     optimizer.maximize(
#         init_points=3,
#         n_iter=10,
#     )    


    



if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(2)
    best_accu = 0

    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
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
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'bayes' / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_path = './save_path/bayes/'#/bayes/pointpillar/'+time.strftime('%m%d-%H%M',time.localtime(time.time()))+'/'

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True) 

    logger = common_utils.create_logger(save_path+'log.txt', rank=cfg.LOCAL_RANK)

    file = open(save_path+'result.txt','w')
    file.write('results\n')
    file.close()

    # head = ''
    # logging.basicConfig(filename='./baseline/pointpillar/log.txt',
    #                     format=head)
    # logger_result = logging.getLogger()
    # logger_result.setLevel(logging.INFO)
    # console = logging.StreamHandler()
    # logging.getLogger('').addHandler(console)

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
    # train_set, train_loader, train_sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_train, workers=args.workers,
    #     logger=logger,
    #     training=True,
    #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
    #     total_epochs=args.epochs
    # )


    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_train, workers=args.workers, logger=logger, training=False
    # )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs


    logger.info('----------------Bayes Optimization----------------')
    # Bounded region of parameter space
    pbounds = {'p1': (0.1, 0.9), 'p2': (0.1, 0.9)}

    optimizer = BayesianOptimization(
        f=opt_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 0.11, 'p2': 0.11},
        lazy=True,
    )

    logger_bayes = JSONLogger(path=save_path+"logs2.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)
    
    
    n = 0
    optimizer.maximize(
        init_points=3,
        n_iter=10,
    )
    print("=======end========")



