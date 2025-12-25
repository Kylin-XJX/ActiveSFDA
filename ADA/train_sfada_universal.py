#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import pdb
import time
import socket
import warnings
import gc

from ADA.utils.common import add_config
from ADA.utils.query import Querier
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from xmuda.common.solver.build import build_optimizer, build_scheduler
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate,validate_by_pselab
from xmuda.models.losses import entropy_loss
# from xmuda.common.config.base import CN


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('--ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--debug', type=bool, default=False, help='debugging?')
    parser.add_argument('--ex_tag', type=str, help='extra tag')
    parser.add_argument('--dataset_trg', type=str,default='NuScenesLidarSegSCN', help='target dataset')
    parser.add_argument('--winsize', type=int, help='size of slide window for smooth')

    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def train(cfg, args, run_name='',ckpt_save_dir = '',tb_dir = ''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    # logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    logger.info('#Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    # logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    logger.info('#Parameters: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build optimizer
    optimizer_2d = build_optimizer(cfg, model_2d,cfg.OPTIMIZER.FACTOR_2D)
    optimizer_3d = build_optimizer(cfg, model_3d,cfg.OPTIMIZER.FACTOR_3D)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=ckpt_save_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    logger.info('weight_path:{}'.format(args.ckpt2d))
    checkpoint_data_2d = checkpointer_2d.load_deal_dual(cfg.MODEL_2D,args.ckpt2d, resume=False,load_4_backbone=False,resume_states=False)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=ckpt_save_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    logger.info('weight_path:{}'.format(args.ckpt3d))
    checkpoint_data_3d = checkpointer_3d.load_deal_dual(cfg.MODEL_3D,args.ckpt3d, resume=False,load_4_backbone=False,resume_states=False)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if tb_dir:
        tb_path = osp.join(tb_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_path)
    else:
        summary_writer = None
    # summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
            
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD

    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None
    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric_val = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None),
        '2d_3d': None
    }
    best_metric_iter_val = {'2d': -1, '3d': -1}

    test_dataloader = build_dataloader(cfg, mode='test', domain='target') if val_period > 0 else None

    best_metric_test = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None),
        '2d_3d': None
    }
    best_metric_iter_test = {'2d': -1, '3d': -1}
    
    best_metric_test_cur = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None),
        '2d_3d': None
    }
    best_metric_iter_test_cur = {'2d': -1, '3d': -1}

    dual_model = (model_2d is not None) and (model_3d is not None)
    metric_modality = ['2d', '3d']
    if dual_model:
        metric_modality = metric_modality + ['2d_3d']
    
    iou_values = {
        '2d': [],
        '3d': [],
        '2d_3d': []
    }

    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    # train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    train_metric_logger = MetricLogger(delimiter='  ')
    val_metric_logger = MetricLogger(delimiter='  ')
    train_val_metric_logger = MetricLogger(delimiter='  ')
    if args.winsize:
        test_metric_logger = MetricLogger(delimiter='  ',window_size=args.winsize)
        logger.info(f"Init MetricLogger with window size = {args.winsize}")
    else:
        test_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()
        train_val_metric_logger.reset()
        # test_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    if cfg.TRAIN.CLASS_WEIGHTS_PL:
        class_weights_pl = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS_PL).cuda()
    else:
        class_weights_pl = None

    logger.info(f"class_weights:{class_weights}")
    logger.info(f"class_weights_pl:{class_weights_pl}")

    setup_train()
    end = time.time()

    train_iter_trg = enumerate(train_dataloader_trg)

    querier = Querier(cfg,run_name,logger)
    load_mask_from_disk = cfg.ADA.load_mask_from_disk
    logger.info(f'load_mask_from_disk:{load_mask_from_disk}')

    update_pselab = cfg.ADA.update_pselab
    queried = False
    for iteration in tqdm(range(start_iteration, max_iteration)):
        # pdb.set_trace()
        #queried = False
        if (not load_mask_from_disk) and querier.query_test(iteration,iou_values):

            if iteration!=0:
                queried = True
            params = {
                'cfg':cfg,
                'dataset':train_dataloader_trg.dataset,
                'model_2d':model_2d,
                'model_3d':model_3d,
                'logger':logger
            }
            if iteration != 0 and update_pselab:
                cur_pselab_save_dir = osp.join(
                    cfg.ADA.pselab_save_dir,
                    run_name)

                if not osp.isdir(cur_pselab_save_dir):
                    logger.info('Make a new directory: {}'.format(cur_pselab_save_dir))
                    os.makedirs(cur_pselab_save_dir,exist_ok=True)

                cur_pselab_save_path = osp.join(
                    cur_pselab_save_dir,
                    f'iter{iteration}_pselab.npy')
                params['pselab_save_path'] = cur_pselab_save_path
                
            else:
                cur_pselab_save_path = None

            querier.query(iteration,params)

            # reload dataloader with new label mask
            cfg.defrost()
            dataset_target_type = cfg.DATASET_TARGET.TYPE
            cfg.DATASET_TARGET[dataset_target_type].label_mask_path = querier.cur_label_mask_save_path
            
            if cur_pselab_save_path:
                cfg.DATASET_TARGET[dataset_target_type].pselab_paths = (cur_pselab_save_path,)
                cfg.DATASET_TARGET[dataset_target_type].new_pselab = True
            cfg.freeze()
            # pdb.set_trace()
            # free memory
            train_dataloader_trg = None
            del train_dataloader_trg
            train_iter_trg = None
            del train_iter_trg
            params = None
            del params
            gc.collect()

            # pdb.set_trace()

            train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=iteration)
            train_iter_trg = enumerate(train_dataloader_trg)
            
            # pdb.set_trace()

        # fetch data_batches for source & target
        _, data_batch_trg = train_iter_trg.__next__()
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            # target
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')
        
        # pdb.set_trace()

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)

        loss_2d = []
        loss_3d = []

        if cfg.TRAIN.XMUDA.lambda_ce_trg > 0:
            
            # Use a small number of annotations for supervision
            masked_seg_label = data_batch_trg['seg_label'].clone()
            # Ignore unlabeled parts
            masked_seg_label[~data_batch_trg['label_mask']] = -100

            seg_loss_trg_2d = cfg.TRAIN.XMUDA.lambda_ce_trg * F.cross_entropy(preds_2d['seg_logit'], masked_seg_label, weight=class_weights)
            seg_loss_trg_3d = cfg.TRAIN.XMUDA.lambda_ce_trg * F.cross_entropy(preds_3d['seg_logit'], masked_seg_label, weight=class_weights)

            # pdb.set_trace()
            train_metric_logger.update(seg_loss_trg_2d=seg_loss_trg_2d,
                                       seg_loss_trg_3d=seg_loss_trg_3d)

            loss_2d.append(seg_loss_trg_2d)
            loss_3d.append(seg_loss_trg_3d)

        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = cfg.TRAIN.XMUDA.lambda_xm_trg * F.kl_div(
                F.log_softmax(seg_logit_2d, dim=1),
                F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                reduction='none').sum(1).mean()
            
            xm_loss_trg_3d = cfg.TRAIN.XMUDA.lambda_xm_trg * F.kl_div(
                F.log_softmax(seg_logit_3d, dim=1),
                F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                reduction='none').sum(1).mean()
            
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(xm_loss_trg_2d)
            loss_3d.append(xm_loss_trg_3d)

        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            # Use a small number of annotations to correct pseudo labels

            # if 'pred_label_ensemble' in data_batch_trg.keys():
            #     pseudo_label_2d = data_batch_trg['pred_label_ensemble'].clone()
            #     pseudo_label_3d = data_batch_trg['pred_label_ensemble'].clone()
            # else:
            #     pseudo_label_2d = data_batch_trg['pseudo_label_2d']
            #     pseudo_label_3d = data_batch_trg['pseudo_label_3d'] 

            pseudo_label_2d = data_batch_trg['pseudo_label_2d']
            pseudo_label_3d = data_batch_trg['pseudo_label_3d']

            if 'label_mask' in data_batch_trg.keys():
                pseudo_label_2d[data_batch_trg['label_mask']] = data_batch_trg['seg_label'][data_batch_trg['label_mask']]

                pseudo_label_3d[data_batch_trg['label_mask']] = data_batch_trg['seg_label'][data_batch_trg['label_mask']]

            # pdb.set_trace()     

            # uni-modal self-training loss with pseudo labels
            pl_loss_trg_2d = cfg.TRAIN.XMUDA.lambda_pl * F.cross_entropy(preds_2d['seg_logit'], pseudo_label_2d,weight=class_weights_pl)
            pl_loss_trg_3d = cfg.TRAIN.XMUDA.lambda_pl * F.cross_entropy(preds_3d['seg_logit'], pseudo_label_3d,weight=class_weights_pl)

            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(pl_loss_trg_2d)
            loss_3d.append(pl_loss_trg_3d)

        if cfg.TRAIN.XMUDA.lambda_minent > 0:
            # MinEnt
            minent_loss_trg_2d = cfg.TRAIN.XMUDA.lambda_minent * entropy_loss(F.softmax(preds_2d['seg_logit'], dim=1))
            minent_loss_trg_3d = cfg.TRAIN.XMUDA.lambda_minent * entropy_loss(F.softmax(preds_3d['seg_logit'], dim=1))
            train_metric_logger.update(minent_loss_trg_2d=minent_loss_trg_2d,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d.append(minent_loss_trg_2d)
            loss_3d.append(minent_loss_trg_3d)

        sum(loss_2d).backward()
        sum(loss_3d).backward()

        optimizer_2d.step()
        optimizer_3d.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr_2d: {lr_2d:.2e}',
                        'lr_3d: {lr_3d:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr_2d=optimizer_2d.param_groups[0]['lr'],
                    lr_3d=optimizer_3d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if querier.auto_query and not queried and val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            
            # ***************************
            # validate using training set
            # ***************************
            logger.info("Validate using training set.")
            start_time_val = time.time()
            setup_validate()

            validate_by_pselab(cfg,
                               model_2d,
                               model_3d,
                               train_dataloader_trg.dataset,
                               train_val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Training Set Val {}  total_time: {:.2f}s'.format(
                cur_iter, train_val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in train_val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

            # log validation results
            for modality in metric_modality:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality + '_pselab'
                # print('train_val_metric_logger.meters',train_val_metric_logger.meters)
                if cur_metric_name in train_val_metric_logger.meters:
                    cur_metric = train_val_metric_logger.meters[cur_metric_name].global_avg
                    # Record each modality mIOU
                    iou_values[modality].append(cur_metric)
            print('iou_values',iou_values)
            # restore training
            setup_train()


        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            # **********************
            # validate using val set
            # **********************
            logger.info("Validate using val set.")
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in metric_modality:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric_val[modality] is None or best_metric_val[modality] < cur_metric:
                        best_metric_val[modality] = cur_metric
                        best_metric_iter_val[modality] = cur_iter
                
            for modality in metric_modality:
                logger.info('Current best val-{}-{} = {:.2f} at iteration {}'.format(
                    modality.upper(),
                    cfg.VAL.METRIC,
                    best_metric_val[modality] * 100,
                    best_metric_iter_val[modality]))
            # restore training
            setup_train()
                
            # ***********************
            # validate using test set
            # ***********************
        if val_period > 0 and (cur_iter % (val_period*2) == 0 or cur_iter == max_iteration):
            logger.info("Validate using test set.")
            start_time_test = time.time()
            setup_validate()
            validate(cfg,
                     model_2d,
                     model_3d,
                     test_dataloader,
                     test_metric_logger)

            epoch_time_test = time.time() - start_time_test
            logger.info('Iteration[{}]-Test {}  total_time: {:.2f}s'.format(
                cur_iter, test_metric_logger.summary_str, epoch_time_test))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in test_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('test/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in metric_modality:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in test_metric_logger.meters:
                    save_ckpt = False
                    # cur_metric = test_metric_logger.meters[cur_metric_name].global_avg
                    cur_metric = test_metric_logger.meters[cur_metric_name].avg
                    if best_metric_test[modality] is None or best_metric_test[modality] < cur_metric:
                        best_metric_test[modality] = cur_metric
                        best_metric_iter_test[modality] = cur_iter
                        save_ckpt = True

                    cur_metric_last = test_metric_logger.meters[cur_metric_name].values[-1]
                    if best_metric_test_cur[modality] is None or best_metric_test_cur[modality] < cur_metric_last:
                        best_metric_test_cur[modality] = cur_metric_last
                        best_metric_iter_test_cur[modality] = cur_iter
                        save_ckpt = True
                    if save_ckpt:
                        logger.info('New Best! Saving ckpt!')
                        checkpoint_data_2d['iteration'] = cur_iter
                        checkpoint_data_2d[best_metric_name] = best_metric_test_cur['2d']
                        checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
                        checkpoint_data_3d['iteration'] = cur_iter
                        checkpoint_data_3d[best_metric_name] = best_metric_test_cur['3d']
                        checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)
                
            for modality in metric_modality:
                logger.info('Current best avg test-{}-{} = {:.2f} at iteration {}'.format(
                    modality.upper(),
                    cfg.VAL.METRIC,
                    best_metric_test[modality] * 100,
                    best_metric_iter_test[modality]))  
            for modality in metric_modality:
                logger.info('Current best test-{}-{} = {:.2f} at iteration {}'.format(
                    modality.upper(),
                    cfg.VAL.METRIC,
                    best_metric_test_cur[modality] * 100,
                    best_metric_iter_test_cur[modality]))              


            # restore training
            setup_train()

        # checkpoint
        # if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
        if cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric_test['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric_test['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)
            
        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

        

    for modality in metric_modality:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(
            modality.upper(),
            cfg.VAL.METRIC,
            best_metric_val[modality] * 100,
            best_metric_iter_val[modality]))
        
    for modality in metric_modality:
        logger.info('Best avg test-{}-{} = {:.2f} at iteration {}'.format(
            modality.upper(),
            cfg.VAL.METRIC,
            best_metric_test[modality] * 100,
            best_metric_iter_test[modality]))
        
    for modality in metric_modality:
        logger.info('Best test-{}-{} = {:.2f} at iteration {}'.format(
            modality.upper(),
            cfg.VAL.METRIC,
            best_metric_test_cur[modality] * 100,
            best_metric_iter_test_cur[modality]))

def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    add_config(cfg,args.dataset_trg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    set_random_seed(cfg.RNG_SEED)
    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    if args.ex_tag:
        run_name = run_name + '.' + args.ex_tag
    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))

        if args.debug:
            output_dir = osp.join(output_dir,'debug')

        log_dir = osp.join(output_dir,'logs/')
        tb_dir = osp.join(output_dir,'tb_logs/')
        ckpt_save_dir = osp.join(output_dir,'ckpts',timestamp+'_ckpt')

        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir,exist_ok=True)
        if not osp.isdir(log_dir):
            warnings.warn('Make a new directory: {}'.format(log_dir))
            os.makedirs(log_dir,exist_ok=True)
        if not osp.isdir(tb_dir):
            warnings.warn('Make a new directory: {}'.format(tb_dir))
            os.makedirs(tb_dir,exist_ok=True)
        if not osp.isdir(ckpt_save_dir):
            warnings.warn('Make a new directory: {}'.format(ckpt_save_dir))
            os.makedirs(ckpt_save_dir,exist_ok=True)

    logger = setup_logger('xmuda', log_dir, comment='train.{:s}'.format(run_name))

    if output_dir:
        logger.info('output dir:{}'.format(output_dir))
        logger.info('log_dir:{}'.format(log_dir))
        logger.info('ckpt_save_dir:{}'.format(ckpt_save_dir))

    logger.info('run_name:{}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg = cfg,
          args = args,
          run_name = run_name,
          ckpt_save_dir=ckpt_save_dir,
          tb_dir=tb_dir)

if __name__ == '__main__':
    main()
