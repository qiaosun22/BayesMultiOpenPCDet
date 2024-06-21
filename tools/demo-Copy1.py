import mayavi.mlab as mlab
import argparse
import glob
from pathlib import Path


import numpy as np
import torch
import os
import matplotlib
# mlab.use('TkAgg')
from pcdet.datasets import build_dataloader# build_dataloader_test, 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
print('Using GPU:' + str(np.argmax(memory_gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')

from pcdet.config import cfg, cfg1, cfg2, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
from noise import add_noise_to_weights
from test import eval_single_ckpt


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file1', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--cfg_file2', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--sigma', type=float, default=0, help='specify the noise model')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file1, cfg1)
    cfg_from_yaml_file(args.cfg_file2, cfg2)

    return args, cfg1, cfg2


def main():
    args, cfg1, cfg2 = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg1.DATA_CONFIG, class_names=cfg1.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # test_set, test_loader, sampler = build_dataloader(
    # dataset_cfg=cfg1.DATA_CONFIG,
    # class_names=cfg1.CLASS_NAMES,
    # batch_size=1,
    # dist=False, workers=1, logger=logger, training=False
    # )

    model1= build_network(model_cfg=cfg1.MODEL, num_class=len(cfg1.CLASS_NAMES), p1=0, p2=0, dataset=demo_dataset)

    model2= build_network(model_cfg=cfg2.MODEL, num_class=len(cfg1.CLASS_NAMES), p1=0, p2=0, dataset=demo_dataset)

    # print('=====len(test_loader)====')
    # print(len(test_loader))
    # print(test_loader[0].shape)
    sigma = [0,0.2,0.4]

    with torch.no_grad():
        print('====data_length===')
        # print(len(demo_dataset))
        for idx, data_dict in enumerate(demo_dataset):
        # for idx, data_dict in enumerate(test_loader):

            data_dict = demo_dataset.collate_batch([data_dict])

            logger.info(f'Visualized sample index: \t{idx + 1}')

            output_path = './pointcloud_results/pointpillar/'+args.ckpt+'/'+str(idx)
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True) 

            np.save(output_path+"/points.npy",data_dict['points'][:, 1:])     
            # np.save(output_path+"/pred_boxes_gt.npy",data_dict['gt_boxes'])       
            # print(data_dict['gt_boxes'].shape)

            for s in sigma:    
                model1.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
                model1.cuda()
                add_noise_to_weights(0, s, model1)
                model1.eval()

                load_data_to_gpu(data_dict)
                pred_dicts, _ = model1.forward(data_dict)

                print((pred_dicts[0]['pred_boxes']).shape[0])

                np.save(output_path+"/pred_boxes-@{}.npy".format(s),pred_dicts[0]['pred_boxes'].cpu().numpy())
                np.save(output_path+"/pred_scores-@{}.npy".format(s),pred_dicts[0]['pred_scores'].cpu().numpy())
                np.save(output_path+"/pred_labels-@{}.npy".format(s),pred_dicts[0]['pred_labels'].cpu().numpy())

   
                model2.load_params_from_file(filename='./bayes/pointpillar/0623-2211/bayes_model-0.15-0.15.pth', logger=logger, to_cpu=True)
                model2.cuda()
                add_noise_to_weights(0, s, model2)                
                model2.eval()

                load_data_to_gpu(data_dict)
                pred_dicts, _ = model2.forward(data_dict)

                print((pred_dicts[0]['pred_boxes']).shape[0])

                np.save(output_path+"/pred_boxes-bayes-@{}.npy".format(s),pred_dicts[0]['pred_boxes'].cpu().numpy())
                np.save(output_path+"/pred_scores-bayes-@{}.npy".format(s),pred_dicts[0]['pred_scores'].cpu().numpy())
                np.save(output_path+"/pred_labels-bayes-@{}.npy".format(s),pred_dicts[0]['pred_labels'].cpu().numpy())
                
            # if idx >=0:
            #     print(data_dict['points'][:, 1:].cpu().numpy())
            #     exit()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
