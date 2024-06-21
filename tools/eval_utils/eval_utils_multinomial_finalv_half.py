import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import datetime


def eval_(cfg, model, dataloader):
    
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    model.eval()

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names, None
        )
        det_annos += annos

    ret_dict = {}

    gt_num_cnt = metric['gt_num']
    
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)

        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    print('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=None
    )

    ret_dict.update(result_dict)

    # ======================3d
    acc1 = float((ret_dict['Car_3d/easy_R40'] + \
                ret_dict['Pedestrian_3d/easy_R40'] + \
                ret_dict['Cyclist_3d/easy_R40'] + \
                ret_dict['Car_3d/moderate_R40'] + \
                ret_dict['Pedestrian_3d/moderate_R40'] + \
                ret_dict['Cyclist_3d/moderate_R40'] + \
                ret_dict['Car_3d/hard_R40'] + \
                ret_dict['Pedestrian_3d/hard_R40'] + \
                ret_dict['Cyclist_3d/hard_R40'])/9)

    save_path = './save_path/'
    
    # ======================bev
    acc2 = float((ret_dict['Car_bev/easy_R40'] + \
                ret_dict['Pedestrian_bev/easy_R40'] + \
                ret_dict['Cyclist_bev/easy_R40'] + \
                ret_dict['Car_bev/moderate_R40'] + \
                ret_dict['Pedestrian_bev/moderate_R40'] + \
                ret_dict['Cyclist_bev/moderate_R40'] + \
                ret_dict['Car_bev/hard_R40'] + \
                ret_dict['Pedestrian_bev/hard_R40'] + \
                ret_dict['Cyclist_bev/hard_R40'])/9)

    return acc1, acc2, ret_dict

    
def eval_simple(p1, p2, p3, file, usability, sigma, 
                n, cfg, model, dataloader, logger, 
                save_path, dist_test=False,
                save_to_file=False, result_dir=None):
    
    # result_dir.mkdir(parents=True, exist_ok=True)

    # if save_to_file:
    #     final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

#     if dist_test:
#         num_gpus = torch.cuda.device_count()
#         print('num_gpus')
#         local_rank = cfg.LOCAL_RANK % num_gpus
#         print('num_gpus')
#         model = torch.nn.parallel.DistributedDataParallel(
#                 model,
#                 device_ids=[local_rank],
#                 broadcast_buffers=False
#         )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
#     start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path= None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

#     if cfg.LOCAL_RANK == 0:
#         progress_bar.close()

#     if dist_test:
#         rank, world_size = common_utils.get_dist_info()
#         det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
#         metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

#     sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
#     logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

#     if cfg.LOCAL_RANK != 0:
#         return 0

    ret_dict = {}
#     if dist_test:
#         for key, val in metric[0].items():
#             for k in range(1, world_size):
#                 metric[0][key] += metric[k][key]
#         metric = metric[0]

    gt_num_cnt = metric['gt_num']
    
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        # logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        # logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=result_dir
    )


    ret_dict.update(result_dict)


    # ======================3d
    acc1 = float((ret_dict['Car_3d/easy_R40'] + \
                ret_dict['Pedestrian_3d/easy_R40'] + \
                ret_dict['Cyclist_3d/easy_R40'] + \
                ret_dict['Car_3d/moderate_R40'] + \
                ret_dict['Pedestrian_3d/moderate_R40'] + \
                ret_dict['Cyclist_3d/moderate_R40'] + \
                ret_dict['Car_3d/hard_R40'] + \
                ret_dict['Pedestrian_3d/hard_R40'] + \
                ret_dict['Cyclist_3d/hard_R40'])/9)

    save_path = './save_path/'
    f = open(save_path+'3d.txt', "a+")

    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                        file, usability, sigma, n, '3d',
                                        ret_dict['Car_3d/easy_R40'],
                                        ret_dict['Car_3d/moderate_R40'],
                                        ret_dict['Car_3d/hard_R40'],
                                        ret_dict['Pedestrian_3d/easy_R40'],
                                        ret_dict['Pedestrian_3d/moderate_R40'],
                                        ret_dict['Pedestrian_3d/hard_R40'],
                                        ret_dict['Cyclist_3d/easy_R40'],
                                        ret_dict['Cyclist_3d/moderate_R40'],
                                        ret_dict['Cyclist_3d/hard_R40'],
                                        (ret_dict['Car_3d/easy_R40'] + ret_dict['Pedestrian_3d/easy_R40'] + ret_dict['Cyclist_3d/easy_R40'])/3,
                                        (ret_dict['Car_3d/moderate_R40'] + ret_dict['Pedestrian_3d/moderate_R40'] + ret_dict['Cyclist_3d/moderate_R40'])/3,
                                        (ret_dict['Car_3d/hard_R40'] + ret_dict['Pedestrian_3d/hard_R40'] + ret_dict['Cyclist_3d/hard_R40'])/3,
                                        acc1, str(datetime.datetime.now()).replace(' ', '-'), 
                                        p1, p2, p3        
                                        ))
    f.close()

    # ======================bev
    acc2 = float((ret_dict['Car_bev/easy_R40'] + \
                ret_dict['Pedestrian_bev/easy_R40'] + \
                ret_dict['Cyclist_bev/easy_R40'] + \
                ret_dict['Car_bev/moderate_R40'] + \
                ret_dict['Pedestrian_bev/moderate_R40'] + \
                ret_dict['Cyclist_bev/moderate_R40'] + \
                ret_dict['Car_bev/hard_R40'] + \
                ret_dict['Pedestrian_bev/hard_R40'] + \
                ret_dict['Cyclist_bev/hard_R40'])/9)

    f = open(save_path+'bev.txt', "a+")

    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                        file, usability, sigma, n, 'bev', 
                                        ret_dict['Car_bev/easy_R40'],
                                        ret_dict['Car_bev/moderate_R40'],
                                        ret_dict['Car_bev/hard_R40'],
                                        ret_dict['Pedestrian_bev/easy_R40'],
                                        ret_dict['Pedestrian_bev/moderate_R40'],
                                        ret_dict['Pedestrian_bev/hard_R40'],
                                        ret_dict['Cyclist_bev/easy_R40'],
                                        ret_dict['Cyclist_bev/moderate_R40'],
                                        ret_dict['Cyclist_bev/hard_R40'],
                                        (ret_dict['Car_bev/easy_R40'] + ret_dict['Pedestrian_bev/easy_R40'] + ret_dict['Cyclist_bev/easy_R40'])/3,
                                        (ret_dict['Car_bev/moderate_R40'] + ret_dict['Pedestrian_bev/moderate_R40'] + ret_dict['Cyclist_bev/moderate_R40'])/3,
                                        (ret_dict['Car_bev/hard_R40'] + ret_dict['Pedestrian_bev/hard_R40'] + ret_dict['Cyclist_bev/hard_R40'])/3,
                                        acc2, str(datetime.datetime.now()).replace(' ', '-'),
                                        p1, p2, p3                                                          
                                        ))
    f.close()

    return acc1, ret_dict



if __name__ == '__main__':
    pass
