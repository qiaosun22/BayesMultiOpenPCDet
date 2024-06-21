import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt, eval_single_ckpt
# from noise import add_noise_to_weights
from hardware_noise.weight_mapping_finalv import weight_mapping as add_noise_to_weights

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
from pcdet.models_multinomial_finalv import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model, model_save
from eval_utils import eval_utils_multinomial_finalv as eval_utils
import numpy as np



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./cfgs/kitti_models/pointpillar_bayes.yaml', \
                        help='specify the config for training')
    # sunqiao/OpenPCDet/tools/cfgs/kitti_models/pointpillar_bayes.yaml
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=32, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='./checkpoint_epoch_33_multinomial.pth', help='checkpoint to start from')
    # ./checkpoint_epoch_80.pth
    parser.add_argument('--pretrained_model', type=str, default=True, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=81, help='max number of saved checkpoint')
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

class Opt():
    def __init__(self, sigma, file):
        self.sigma = sigma
        self.file = file
        
    def opt_function(self, pp1, pp2, pp3, pp4):
        
        global p1
        global p2
        global p3
        global p4
        
        p1 = pp1
        p2 = pp2
        p3 = pp3
        p4 = pp4
        
        print("=============")
        print(p1, p2, p3, p4)
        print("=============")

        global best_accu
        
        with_training = False
        if with_training == True:
            # p1 = round(p1, 2)
            # p2 = round(p2, 2)
            # p3 = round(p3, 2)
            # p4 = round(p4, 2)
            
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
                                p3=p3, 
                                p4=p4, 
                                dataset=train_set)
            print(model.state_dict())

            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()

            optimizer = build_optimizer(model, cfg.OPTIMIZATION)

            # # load checkpoint if it is possible
            start_epoch = it = 0
            last_epoch = -1
            if args.pretrained_model is True:
                model.load_params_from_file(filename=args.ckpt, to_cpu=dist, logger=logger)


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


        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), p1=0.11, p2=0.11, p3=0.11, p4=0.11, dataset=test_set)

        optimizer = build_optimizer(model, cfg.OPTIMIZATION)

        if dist_train: 
            model = model.module

        if args.pretrained_model is True:
            model.load_params_from_file(filename=args.ckpt, to_cpu=dist, logger=logger)

        # model.load_params_from_file(filename='./checkpoint_epoch_80.pth', logger=logger, to_cpu=dist_train)
        model.cuda()

        ckpt_pth = save_path+'bayes_model-{}-{}'.format(p1, p2, p3, p4)
        ckpt_name = ckpt_pth+'.pth'

        if cfg.LOCAL_RANK == 0:
            model_save(model, ckpt_pth, optimizer, args.epochs, args.epochs)

        logger.info('**********************End training**********************')

        time.sleep(30)


        sigma = self.sigma
        f = open(save_path+'result.txt', "a+")
        # f.write('----------------Noise-{}-evaluate----------------'.format(sigma))
        f.write('----------------{}-{}-{}-{}---------------\n'.format(p1, p2, p3, p4))
        f.close()

        logger.info('----------------Noise-{}-evaluate----------------'.format(file))

        model = add_noise_to_weights(self.file, 0, sigma, model, 'cuda')
        global n
        n += 1

        acc1, ret_dict = eval_utils.eval_simple(p1, p2, p3, p4, sigma, n, cfg, model, test_loader, logger, save_path, dist_test=dist_train, save_to_file=args.save_to_file, result_dir=eval_output_dir)
        print("----------")
        print(acc1)
        print("----------")


        logger.info('**********************End evaluation**********************')

        return acc1




if __name__ == '__main__':

    torch.cuda.set_device(0)
    # best_accu = 0

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


    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs
    
    
    print("=============")
    p1 = 0.23
    p2 = 0.14
    p3 = 0.68
    p4 = 0.36
    
    print(p1, p2, p3, p4)
    print("=============")
    

    '''Bayesian Training'''
    bayes_train = False
    if bayes_train:
        logger.info('----------------Start Bayes Optimization----------------')
        for f_name in os.listdir('hardware_noise'):

            opt= Opt(sigma, file)
            opt_function = opt.opt_function

            # Bounded region of parameter space
            pbounds = {'p1': (0.1, 0.9), 'p2': (0.1, 0.9), 'p3': (0.1, 0.9), 'p4': (0.1, 0.9)}


            optimizer_bayes = BayesianOptimization(
                f=opt_function,
                pbounds=pbounds,
                verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                random_state=1,
            )
            optimizer_bayes.probe(
                params={'p1': 0.11, 'p2': 0.11, 'p3': 0.11, 'p4': 0.11},
                lazy=True,
            )

            logger_bayes = JSONLogger(path=save_path+"logs2.json")
            optimizer_bayes.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)

            n = 0
            optimizer_bayes.maximize(
                init_points=3,
                n_iter=10,
            )
        logger.info('----------------End Bayes Optimization----------------')
            
            
    '''Test on Noises'''
    
    logger.info('**********************Start evaluation**********************')    
    test_set, test_loader, sampler = build_dataloader(
                                    dataset_cfg=cfg.DATA_CONFIG,
                                    class_names=cfg.CLASS_NAMES,
                                    batch_size=args.batch_size,
                                    dist=dist_train, 
                                    workers=args.workers, 
                                    logger=logger,
                                    training=False
                                )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), p1=0.11, p2=0.11, p3=0.11, p4=0.11, dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, to_cpu=dist, logger=logger)
    
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    model.cuda()

    ckpt_pth = save_path+'bayes_model-{}-{}-{}-{}'.format(p1, p2, p3, p4)
    ckpt_name = ckpt_pth+'.pth'
    model_save(model, ckpt_pth, optimizer, args.epochs, args.epochs)

    time.sleep(1)

    hw_data_files = sorted(os.listdir('./hardware_noise/hardware_data/'))
    file2ap_dict = {}
    N = 10
    for f_name in sorted(hw_data_files):
        if f_name.endswith('xlsx'):
            file2ap_dict[f_name] = {}
            # print(f_name)
            for n in range(N):
                print('file:{}, evaluate-{}'.format(f_name, n))
                model.load_params_from_file(filename=args.ckpt, to_cpu=dist, logger=logger)
                add_noise_to_weights('./hardware_noise/hardware_data/'+f_name, model, device='cuda')

                acc1, ret_dict = eval_utils.eval_simple(args.ckpt, p1, p2, p3, f_name, n, cfg, model, test_loader, logger, save_path=None, dist_test=dist_train,
                                                save_to_file=args.save_to_file, result_dir=eval_output_dir
                                            )                
                print(ret_dict)
                file2ap_dict[f_name][n] = ret_dict
                
                
    logger.info('**********************End evaluation**********************')    


    print("=======end========")
