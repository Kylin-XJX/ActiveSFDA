import pdb
import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from xmuda.data.collate import get_collate_scn
from xmuda.data.utils.evaluate import Evaluator
from xmuda.common.utils.torch_util import worker_init_fn
from torch.utils.data.dataloader import DataLoader

from utils.utils import get_mask_prompt,mask_to_boxes


def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    dual_model = (model_2d is not None) and (model_3d is not None)

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names) if model_2d else None
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if dual_model else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch) if model_2d else None
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy() if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if dual_model else None

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if dual_model else None

                # evaluate
                if model_2d:
                    evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                if dual_model:
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    if model_2d:
                        assert np.all(pred_label_2d >= 0)
                    if model_3d:
                        assert np.all(pred_label_3d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx] if model_2d else None
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy() if model_2d else None,
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8)  if model_2d else None,
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label']) if model_2d else None
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            if seg_loss_2d is not None:
                val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        eval_list = []
        if evaluator_2d is not None:
            val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
            eval_list.append(('2D', evaluator_2d))
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
            eval_list.append(('3D', evaluator_3d))
        if dual_model:
            val_metric_logger.update(seg_iou_2d_3d=evaluator_ensemble.overall_iou)
            eval_list.append(('2D+3D', evaluator_ensemble))
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy: {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))

def validate_by_pselab(cfg,
                       model_2d,
                       model_3d,
                       dataset,
                       val_metric_logger,
                       pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Training Set Validation')

    dual_model = (model_2d is not None) and (model_3d is not None)

    # evaluator

    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )

    class_names = dataset.class_names

    evaluator_2d_pselab = Evaluator(class_names) if model_2d else None
    evaluator_3d_pselab = Evaluator(class_names) if model_3d else None
    evaluator_ensemble_pselab = Evaluator(class_names) if dual_model else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(val_loader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch) if model_2d else None
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy() if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if dual_model else None

            # loop over batch
            left_idx = 0
            for batch_ind in range(len(data_batch['img_indices'])):
                
                # right_idx = left_idx + curr_points_idx.sum()
                right_idx = left_idx + len(data_batch['img_indices'][batch_ind])
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if dual_model else None

                curr_seg_label = data_batch['seg_label'][left_idx:right_idx]
                curr_pse_label_2d = data_batch['pseudo_label_2d'][left_idx:right_idx]
                curr_pse_label_3d = data_batch['pseudo_label_3d'][left_idx:right_idx]
                curr_pse_label_ensemble = data_batch['pseudo_label_ensemble'][left_idx:right_idx]

                if model_2d:
                    evaluator_2d_pselab.update(pred_label_2d, curr_pse_label_2d)
                if model_3d:
                    evaluator_3d_pselab.update(pred_label_3d, curr_pse_label_3d)
                if dual_model:
                    evaluator_ensemble_pselab.update(pred_label_ensemble, curr_pse_label_ensemble)

                left_idx = right_idx

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(val_loader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        eval_list_pselab = []
        if evaluator_2d_pselab is not None:
            val_metric_logger.update(seg_iou_2d_pselab=evaluator_2d_pselab.overall_iou)
            eval_list_pselab.append(('2D', evaluator_2d_pselab))
        if evaluator_3d_pselab is not None:
            val_metric_logger.update(seg_iou_3d_pselab=evaluator_3d_pselab.overall_iou)
            eval_list_pselab.append(('3D', evaluator_3d_pselab))
        if dual_model:
            val_metric_logger.update(seg_iou_2d_3d_pselab=evaluator_ensemble_pselab.overall_iou)
            eval_list_pselab.append(('2D+3D', evaluator_ensemble_pselab))

        for modality, evaluator in eval_list_pselab:
            logger.info('{} overall accuracy (pselab): {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU (pselab): {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU (pselab).\n{}'.format(modality, evaluator.print_table()))

def validate_by_pselab_seg_pse(cfg,
                       model_2d,
                       model_3d,
                       dataset,
                       val_metric_logger,
                       pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Training Set Validation')

    dual_model = (model_2d is not None) and (model_3d is not None)

    # evaluator

    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )

    class_names = dataset.class_names
    evaluator_2d_seglab = Evaluator(class_names) if model_2d else None
    evaluator_3d_seglab = Evaluator(class_names) if model_3d else None
    evaluator_ensemble_seglab = Evaluator(class_names) if dual_model else None

    evaluator_2d_pselab = Evaluator(class_names) if model_2d else None
    evaluator_3d_pselab = Evaluator(class_names) if model_3d else None
    evaluator_ensemble_pselab = Evaluator(class_names) if dual_model else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(val_loader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch) if model_2d else None
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy() if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if dual_model else None

            # get original point cloud from before voxelization
            # seg_label = data_batch['orig_seg_label']
            # points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(data_batch['img_indices'])):
                # curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                # assert np.all(curr_points_idx)

                # curr_seg_label = seg_label[batch_ind]
                
                # right_idx = left_idx + curr_points_idx.sum()
                right_idx = left_idx + len(data_batch['img_indices'][batch_ind])
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if dual_model else None

                curr_seg_label = data_batch['seg_label'][left_idx:right_idx]
                curr_pse_label_2d = data_batch['pseudo_label_2d'][left_idx:right_idx]
                curr_pse_label_3d = data_batch['pseudo_label_3d'][left_idx:right_idx]
                curr_pse_label_ensemble = data_batch['pseudo_label_ensemble'][left_idx:right_idx]

                # evaluate
                if model_2d:
                    evaluator_2d_seglab.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d_seglab.update(pred_label_3d, curr_seg_label)
                if dual_model:
                    evaluator_ensemble_seglab.update(pred_label_ensemble, curr_seg_label)

                if model_2d:
                    evaluator_2d_pselab.update(pred_label_2d, curr_pse_label_2d)
                if model_3d:
                    evaluator_3d_pselab.update(pred_label_3d, curr_pse_label_3d)
                if dual_model:
                    evaluator_ensemble_pselab.update(pred_label_ensemble, curr_pse_label_ensemble)

                left_idx = right_idx

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(val_loader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        eval_list_seglab = []
        if evaluator_2d_seglab is not None:
            val_metric_logger.update(seg_iou_2d_seglab=evaluator_2d_seglab.overall_iou)
            eval_list_seglab.append(('2D', evaluator_2d_seglab))
        if evaluator_3d_seglab is not None:
            val_metric_logger.update(seg_iou_3d_seglab=evaluator_3d_seglab.overall_iou)
            eval_list_seglab.append(('3D', evaluator_3d_seglab))
        if dual_model:
            val_metric_logger.update(seg_iou_2d_3d_seglab=evaluator_ensemble_seglab.overall_iou)
            eval_list_seglab.append(('2D+3D', evaluator_ensemble_seglab))

        for modality, evaluator in eval_list_seglab:
            logger.info('{} overall accuracy (seglab): {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU (seglab): {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU (seglab).\n{}'.format(modality, evaluator.print_table()))

        eval_list_pselab = []
        if evaluator_2d_pselab is not None:
            val_metric_logger.update(seg_iou_2d_pselab=evaluator_2d_pselab.overall_iou)
            eval_list_pselab.append(('2D', evaluator_2d_pselab))
        if evaluator_3d_pselab is not None:
            val_metric_logger.update(seg_iou_3d_pselab=evaluator_3d_pselab.overall_iou)
            eval_list_pselab.append(('3D', evaluator_3d_pselab))
        if dual_model:
            val_metric_logger.update(seg_iou_2d_3d_pselab=evaluator_ensemble_pselab.overall_iou)
            eval_list_pselab.append(('2D+3D', evaluator_ensemble_pselab))

        for modality, evaluator in eval_list_pselab:
            logger.info('{} overall accuracy (pselab): {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU (pselab): {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU (pselab).\n{}'.format(modality, evaluator.print_table()))