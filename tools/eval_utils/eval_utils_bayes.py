import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import datetime


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)

    print(result_dict)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_simple(p1, p2, s, n, cfg, model, dataloader, logger, save_path, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if dist_test:
        num_gpus = torch.cuda.device_count()
        print('num_gpus')
        local_rank = cfg.LOCAL_RANK % num_gpus
        print('num_gpus')
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return 0

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    # print(6)
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # with open(result_dir / 'result.pkl', 'wb') as f:
    #     pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    # logger.info(result_str)

    ret_dict.update(result_dict)

    print(ret_dict)


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

    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(s, n, '3d',
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
                                        p1, p2
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

    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(s, n, 'bev', 
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
                                        p1, p2                                                          
                                        ))
    f.close()

    return acc1

if __name__ == '__main__':
    pass
