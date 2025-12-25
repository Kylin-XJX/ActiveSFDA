import copy
import random
import time
import os
import os.path as osp
from cuml.neighbors import NearestNeighbors
import pandas as pd
import torch
from ADA.utils.common import add_config, check_logger, get_sam_masks,get_cur_mask,calculate_proportions
from SAM.segment_anything import sam_model_registry,SamPredictor,SamAutomaticMaskGenerator
import numpy as np
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.data.collate import get_collate_scn
from xmuda.common.utils.torch_util import worker_init_fn
from xmuda.data.build import build_dataloader
from xmuda.models.build import build_model_2d, build_model_3d

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pdb
import os.path as osp
from argparse import Namespace
import torch.nn.functional as F
from scipy.optimize import curve_fit
from skimage.segmentation import slic

# from ADA.Annotator.sampler import get_point_in_voxel
# from torch_scatter import scatter_mean
from sklearn.cluster import KMeans
from math import ceil

class Querier():
    def __init__(self,cfg,run_name,logger) -> None:

        self.logger = logger

        self.query_iters_percent = cfg.ADA.query_iters
        self.max_iteration = cfg.SCHEDULER.MAX_ITERATION
        self.query_iters = [int(self.max_iteration * perc) for perc in self.query_iters_percent]

        self.label_mask_save_dir = cfg.ADA.save_dir
        self.label_mask_save_dir = osp.join(self.label_mask_save_dir,run_name)
        if not osp.isdir(self.label_mask_save_dir):
            self.logger.warn('Make a new directory: {}'.format(self.label_mask_save_dir))
            os.makedirs(self.label_mask_save_dir,exist_ok=True)
        self.label_budget = cfg.ADA.budget
        if len(self.query_iters) == 0:
            self.ratio_per_query = 0
        else:
            self.ratio_per_query = self.label_budget / len(self.query_iters)
        self.label_budget_used = 0

        self.query_function_name = cfg.ADA.query_function_name

        self.auto_query = -1.0 in self.query_iters_percent
        self.r_threshold = cfg.ADA.r_threshold
        self.max_manual_iter = max(self.query_iters)
        self.pre_miou_list_len = 0

        self.logger.info("Querier init complete.")
        self.logger.info(f'query_iters_percent:{self.query_iters_percent}\nquery_iters:{self.query_iters}\nlabel_budget:{self.label_budget}\nratio_per_query:{self.ratio_per_query}\nquery_function_name:{self.query_function_name}\nauto_query:{self.auto_query}\nmax_manual_iter:{self.max_manual_iter}')

    def query_test(self,iteration,miou_dict = None):
        if self.label_budget_used < self.label_budget:
            if iteration <= self.max_manual_iter:
                return iteration in self.query_iters
            else:
                return self.auto_query and self.fit_and_decide(miou_dict)
        else:
            return False
    
    def fit_and_decide(self,miou_dict:dict):
        if miou_dict is None:
            self.logger.warn("miou_dict should not be None if auto_query is True")
            return False
        
        assert len(miou_dict['2d']) == len(miou_dict['3d']) == len(miou_dict['2d_3d']),"mIOU lists length inequal."
        miou_cnt = len(miou_dict['2d'])

        if miou_cnt <= self.pre_miou_list_len:
            return False
        self.pre_miou_list_len = miou_cnt
        
        if miou_cnt < 5:
            self.logger.warn(f"{miou_cnt} points, Too few points")
            return False

        self.logger.info(f"Fitting miou curve using {miou_cnt} points")
        self.logger.info(f"miou_2d:{miou_dict['2d']}")
        self.logger.info(f"miou_3d:{miou_dict['3d']}")
        self.logger.info(f"miou_2d3d:{miou_dict['2d_3d']}")

        def curve_func(x, a, b, c):
            # Add 1 to avoid domain problems with logarithmic functions
            return b * np.log(a*x + 1) + c  
            
        def derivation(x, a, b, c):
            x = x + 1e-6  # numerical robustness
            return (a*b) / (a*x + 1)
        
        def relative_change(t,popt):
            return abs(abs(derivation(t, *popt)) - abs(derivation(1, *popt))) / abs(derivation(1, *popt))

        consensus = 0
        for modality, values in miou_dict.items():
            x = np.arange(miou_cnt)
            y = values
            # print(y)
            # fitting curve
            try:
                popt, _ = curve_fit(curve_func,x, y,sigma=np.geomspace(1, .1, miou_cnt),absolute_sigma=True,maxfev = 10000)

                for t in x[5:]:
                    rc = relative_change(t,popt)
                    if rc > self.r_threshold:
                        self.logger.info(f"{modality} need query.")
                        consensus += 1
                        # break
            except:
                print(f"{modality} fit fail.")

        # If two or more modalities consider a query necessary, return T; otherwise, return F
        self.logger.info(f"consensus : {consensus}.")
        if consensus >= 2:
            self.logger.info("Decided to query.")
            return True
        else:
            return False

    def query(self,iteration,params):
        self.logger.info(f'iteration {iteration},querying...')
        self.label_budget_used += self.ratio_per_query
        self.cur_label_mask_save_path = osp.join(
            self.label_mask_save_dir,
            f'iter{iteration}_{self.label_budget_used}.npy')
        self.logger.info(f'cur_label_mask_save_path:{self.cur_label_mask_save_path}')
        query_function = globals()[self.query_function_name]
        self.logger.info(f"Querying by {self.query_function_name}")
        query_function(
            ratio = self.ratio_per_query,
            save_path = self.cur_label_mask_save_path,
            **params
        )

def get_random_label_mask(cfg,dataset,model_2d,model_3d,ratio,save_path,logger):
    mylog = check_logger(logger)
    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )
    label_mask = get_cur_mask(cfg,val_loader,logger)

    for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        cur_labeled_mask = copy.deepcopy(label_mask[idx]) 
        unlabel_idx = np.arange(len(cur_labeled_mask))[~cur_labeled_mask]
        cnt_to_query = int((ratio/100.0) * (len(cur_labeled_mask)))
        selected_idx = random.sample(list(unlabel_idx),cnt_to_query)
        label_mask[idx][selected_idx] = True

    # pdb.set_trace()
    
    np.save(save_path,label_mask)
    mylog(f'save label mask to {save_path} !')

def get_label_mask_by_confidence(cfg,dataset,model_2d,model_3d,ratio,save_path,logger):
    mylog = check_logger(logger)
    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )
    label_mask = get_cur_mask(cfg,val_loader,logger)
    # val_iter = enumerate(val_loader)
    # pdb.set_trace()
    with torch.no_grad():
        for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            # if idx>100:
            #     break
            # idx,data_batch = val_iter.__next__()
            data_batch['x'][1] = data_batch['x'][1].cuda()
            # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch['img'] = data_batch['img'].cuda()

            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch)

            pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            pred_probs_2d = pred_probs_2d.detach().cpu().numpy()

            pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)
            pred_probs_3d = pred_probs_3d.detach().cpu().numpy()

            mix_probs = 0.5*(pred_probs_2d + pred_probs_3d)
            max_mix_probs = np.max(mix_probs,axis=1)

            cur_labeled_mask = copy.deepcopy(label_mask[idx]) 
            unlabel_idx = np.arange(len(cur_labeled_mask))[~cur_labeled_mask]
            unlabel_max_mix_probs = max_mix_probs[~cur_labeled_mask]
            sorted_indexes = np.argsort(unlabel_max_mix_probs)

            cnt_to_query = int((ratio/100.0) * (len(cur_labeled_mask)))
            ori_lowest_unlabel_idx = unlabel_idx[sorted_indexes[:cnt_to_query]]

            label_mask[idx][ori_lowest_unlabel_idx] = True

        # pdb.set_trace()
        
        np.save(save_path,label_mask)
        mylog(f'save label mask to {save_path} !')

def get_label_mask_by_pred_ent(cfg,dataset,model_2d,model_3d,ratio,save_path,logger):
    mylog = check_logger(logger)
    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )
    label_mask = get_cur_mask(cfg,val_loader,logger)
    # val_iter = enumerate(val_loader)
    # pdb.set_trace()
    with torch.no_grad():
        for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            # if idx>100:
            #     break
            # idx,data_batch = val_iter.__next__()
            data_batch['x'][1] = data_batch['x'][1].cuda()
            # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch['img'] = data_batch['img'].cuda()

            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch)

            pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            pred_probs_2d = pred_probs_2d.detach().cpu().numpy()

            pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)
            pred_probs_3d = pred_probs_3d.detach().cpu().numpy()

            entropy_2d = -np.sum(pred_probs_2d * np.log2(pred_probs_2d + 1e-10),axis=1)
            entropy_3d = -np.sum(pred_probs_3d * np.log2(pred_probs_3d + 1e-10),axis=1)

            mix_entropy = entropy_2d + entropy_3d

            # mix_probs = 0.5*(pred_probs_2d + pred_probs_3d)
            # max_mix_probs = np.max(mix_probs,axis=1)

            cur_labeled_mask = copy.deepcopy(label_mask[idx]) 
            unlabel_idx = np.arange(len(cur_labeled_mask))[~cur_labeled_mask]
            unlabel_mix_entropy = mix_entropy[~cur_labeled_mask]
            sorted_indexes = np.argsort(unlabel_mix_entropy)
            descending_sorted_indexes = sorted_indexes[::-1]

            cnt_to_query = int((ratio/100.0) * (len(cur_labeled_mask)))
            ori_lowest_unlabel_idx = unlabel_idx[descending_sorted_indexes[:cnt_to_query]]

            label_mask[idx][ori_lowest_unlabel_idx] = True
            # pdb.set_trace()

        # pdb.set_trace()
        
        np.save(save_path,label_mask)
        mylog(f'save label mask to {save_path} !')

def get_label_mask_by_pred_margin(cfg,dataset,model_2d,model_3d,ratio,save_path,logger):
    mylog = check_logger(logger)
    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )
    label_mask = get_cur_mask(cfg,val_loader,logger)
    # val_iter = enumerate(val_loader)
    # pdb.set_trace()
    with torch.no_grad():
        for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            # if idx>100:
            #     break
            # idx,data_batch = val_iter.__next__()
            data_batch['x'][1] = data_batch['x'][1].cuda()
            # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch['img'] = data_batch['img'].cuda()

            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch)

            pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            pred_probs_2d = pred_probs_2d.detach()
            top2_probs_2d, _ = torch.topk(pred_probs_2d, 2, dim=1, largest=True, sorted=True)
            top2_probs_2d = top2_probs_2d.cpu().numpy()
            margin_2d = top2_probs_2d[:,0] - top2_probs_2d[:,1]

            pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)
            pred_probs_3d = pred_probs_3d.detach()
            top2_probs_3d, _ = torch.topk(pred_probs_3d, 2, dim=1, largest=True, sorted=True)
            top2_probs_3d = top2_probs_3d.cpu().numpy()
            margin_3d = top2_probs_3d[:,0] - top2_probs_3d[:,1]

            mix_margin = margin_2d + margin_3d

            # mix_probs = 0.5*(pred_probs_2d + pred_probs_3d)
            # max_mix_probs = np.max(mix_probs,axis=1)

            cur_labeled_mask = copy.deepcopy(label_mask[idx]) 
            unlabel_idx = np.arange(len(cur_labeled_mask))[~cur_labeled_mask]
            unlabel_mix_margin = mix_margin[~cur_labeled_mask]
            sorted_indexes = np.argsort(unlabel_mix_margin)

            cnt_to_query = int((ratio/100.0) * (len(cur_labeled_mask)))
            ori_lowest_unlabel_idx = unlabel_idx[sorted_indexes[:cnt_to_query]]

            label_mask[idx][ori_lowest_unlabel_idx] = True
            # pdb.set_trace()
        # pdb.set_trace()
        
        np.save(save_path,label_mask)
        mylog(f'save label mask to {save_path} !')

def get_label_mask_by_feature_entropy_sam(cfg,dataset,model_2d,model_3d,ratio,save_path,logger,pselab_save_path = None):
    mylog = check_logger(logger)

    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )
    label_mask = get_cur_mask(cfg,val_loader,logger)
    # pdb.set_trace()
    k = cfg.ADA.n_neighbors
    
    mylog('building sam ...')

    sam_model = sam_model_registry[cfg.SAM.MODEL_TYPE](checkpoint=cfg.SAM.CKPT)
    sam_model = sam_model.to(device='cuda')
    predictor = SamPredictor(sam_model)
    delete_road_mask = cfg.SAM.delete_road_mask

    mylog(f"batch_size for query= {cfg.ADA.batch_size},k = {cfg.ADA.n_neighbors},delete_road_mask = {cfg.SAM.delete_road_mask}")

    pselab_data_list = []
    mask_cnt = 0.0
    
    with torch.no_grad():
        for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            # idx,data_batch = val_iter.__next__()
            data_batch['x'][1] = data_batch['x'][1].cuda()
            # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch['img'] = data_batch['img'].cuda()

            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch)

            pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            pred_probs_2d = pred_probs_2d.detach().cpu().numpy()

            pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)
            pred_probs_3d = pred_probs_3d.detach().cpu().numpy()

            pred_label_ensemble = (pred_probs_2d + pred_probs_3d).argmax(1)
            pred_label_2d = pred_probs_2d.argmax(1)
            pred_label_3d = pred_probs_3d.argmax(1)
            #*****************************
            # get knn mask
            #*****************************
            features_2d = preds_2d['feats'].cpu().numpy()
            features_3d = preds_3d['feats'].cpu().numpy()

            # The shape of features should be (n_samples, n_features), where n_samples is the number of features
            # Build nearest neighbor object
            # Select K value

            nbrs_2d = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features_2d)
            nbrs_3d = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features_3d)

            distances_2d, indices_2d = nbrs_2d.kneighbors(features_2d)
            distances_3d, indices_3d = nbrs_3d.kneighbors(features_3d)

            # neighbors_label_2d = data_batch['pseudo_label_ensemble'].numpy()[indices_2d]
            # neighbors_label_3d = data_batch['pseudo_label_ensemble'].numpy()[indices_3d]
            neighbors_label_2d = pred_label_ensemble[indices_2d]
            neighbors_label_3d = pred_label_ensemble[indices_3d]

            neighbors_cls_portion_2d = np.array(list(map(lambda row: calculate_proportions(row,cfg), neighbors_label_2d)))

            neighbors_cls_portion_3d = np.array(list(map(lambda row: calculate_proportions(row,cfg), neighbors_label_3d)))

            entropy_2d = -np.sum(neighbors_cls_portion_2d * np.log2(neighbors_cls_portion_2d + 1e-10),axis=1)
            entropy_3d = -np.sum(neighbors_cls_portion_3d * np.log2(neighbors_cls_portion_3d + 1e-10),axis=1)

            entropy_mix = entropy_2d + entropy_3d

            # pdb.set_trace()

            #*****************************
            # get sam mask
            #*****************************
            cur_pt_img = data_batch['img_indices'][0]
            dilated_masks = get_sam_masks(cfg,predictor,pred_probs_2d,pred_probs_3d,data_batch,delete_road_mask=delete_road_mask)
            all_pt_cnt = len(cur_pt_img)
            all_masked_px_cnt = np.sum(dilated_masks)
            idx_to_label = []
            mask_cnt+=len(dilated_masks)
            for mask in dilated_masks:
                cur_labeled_mask = copy.deepcopy(label_mask[idx])
                
                # How many points can be selected for the current mask
                cnt_to_select_cur_mask = (ratio / 100.0) * all_pt_cnt * (np.sum(mask) / all_masked_px_cnt)
                cnt_to_select_cur_mask = int(cnt_to_select_cur_mask)
                if cnt_to_select_cur_mask == 0:
                    cnt_to_select_cur_mask = 1
                # print(cnt_to_select_cur_mask)
                # The index (T \ F) of unmarked points in the current mask, with a length consistent with the number of points
                in_mask_unlabel_tfmask = mask[cur_pt_img[:,0],cur_pt_img[:,1]] & (~cur_labeled_mask)

                # The entropy corresponding to the points in the current mask
                in_mask_ent = entropy_mix[in_mask_unlabel_tfmask]
                in_mask_ent_ori_idx = np.arange(all_pt_cnt)[in_mask_unlabel_tfmask]
                sorted_ent_indexes = np.argsort(in_mask_ent)
                descending_sorted_ent_indexes = sorted_ent_indexes[::-1]
                idx_to_label.extend(in_mask_ent_ori_idx[descending_sorted_ent_indexes[:cnt_to_select_cur_mask]])

            # pdb.set_trace()
            label_mask[idx][idx_to_label] = True
            if pselab_save_path:
                pseudo_label_dict = {
                    'probs_2d': np.max(pred_probs_2d,axis=1),
                    # 'ori_probs_2d':pred_probs_2d,
                    'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                    'probs_3d': np.max(pred_probs_3d,axis=1),
                    # 'ori_probs_3d':pred_probs_3d,
                    'pseudo_label_3d': pred_label_3d.astype(np.uint8),
                    'pseudo_label_ensemble':pred_label_ensemble.astype(np.uint8),
                }
                pselab_data_list.append(pseudo_label_dict)

        # pdb.set_trace()
        
        np.save(save_path,label_mask)
        mylog(f'save label mask to {save_path} !')
        mylog(f"avg mask cnt per img : {mask_cnt/len(label_mask)}")
        if pselab_save_path:
            np.save(pselab_save_path, pselab_data_list)
            mylog(f'save pseduo label to {pselab_save_path} !')


def get_label_mask_by_feature_entropy_vanilla_sam(cfg,dataset,model_2d,model_3d,ratio,save_path,logger,pselab_save_path = None):
    mylog = check_logger(logger)

    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )
    label_mask = get_cur_mask(cfg,val_loader,logger)
    # pdb.set_trace()
    k = cfg.ADA.n_neighbors
    
    mylog('building sam ...')

    sam_model = sam_model_registry[cfg.SAM.MODEL_TYPE](checkpoint=cfg.SAM.CKPT)
    sam_model = sam_model.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    # predictor = SamPredictor(sam_model)

    delete_road_mask = cfg.SAM.delete_road_mask

    mylog(f"batch_size for query= {cfg.ADA.batch_size},k = {cfg.ADA.n_neighbors},delete_road_mask = {cfg.SAM.delete_road_mask}")

    pselab_data_list = []
    
    with torch.no_grad():
        for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            # idx,data_batch = val_iter.__next__()
            data_batch['x'][1] = data_batch['x'][1].cuda()
            # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch['img'] = data_batch['img'].cuda()

            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch)

            pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            pred_probs_2d = pred_probs_2d.detach().cpu().numpy()

            pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)
            pred_probs_3d = pred_probs_3d.detach().cpu().numpy()

            pred_label_ensemble = (pred_probs_2d + pred_probs_3d).argmax(1)
            pred_label_2d = pred_probs_2d.argmax(1)
            pred_label_3d = pred_probs_3d.argmax(1)
            #*****************************
            # get knn mask
            #*****************************
            features_2d = preds_2d['feats'].cpu().numpy()
            features_3d = preds_3d['feats'].cpu().numpy()

            nbrs_2d = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features_2d)
            nbrs_3d = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features_3d)

            distances_2d, indices_2d = nbrs_2d.kneighbors(features_2d)
            distances_3d, indices_3d = nbrs_3d.kneighbors(features_3d)

            # neighbors_label_2d = data_batch['pseudo_label_ensemble'].numpy()[indices_2d]
            # neighbors_label_3d = data_batch['pseudo_label_ensemble'].numpy()[indices_3d]
            neighbors_label_2d = pred_label_ensemble[indices_2d]
            neighbors_label_3d = pred_label_ensemble[indices_3d]

            neighbors_cls_portion_2d = np.array(list(map(lambda row: calculate_proportions(row,cfg), neighbors_label_2d)))

            neighbors_cls_portion_3d = np.array(list(map(lambda row: calculate_proportions(row,cfg), neighbors_label_3d)))

            entropy_2d = -np.sum(neighbors_cls_portion_2d * np.log2(neighbors_cls_portion_2d + 1e-10),axis=1)
            entropy_3d = -np.sum(neighbors_cls_portion_3d * np.log2(neighbors_cls_portion_3d + 1e-10),axis=1)

            entropy_mix = entropy_2d + entropy_3d

            # pdb.set_trace()

            #*****************************
            # get sam mask
            #*****************************
            cur_pt_img = data_batch['img_indices'][0]
            # dilated_masks = get_sam_masks(cfg,predictor,pred_probs_2d,pred_probs_3d,data_batch,delete_road_mask=delete_road_mask)
            image_np = data_batch['img'][0].permute(1, 2, 0).cpu().numpy()
            image_np_int8 = (image_np * 255).astype(np.uint8)
            auto_anns_dict = mask_generator.generate(image_np_int8)
            auto_masks = []
            for ann in auto_anns_dict:
                auto_masks.append(ann['segmentation'])

            all_pt_cnt = len(cur_pt_img)
            all_masked_px_cnt = np.sum(auto_masks)
            idx_to_label = []

            for mask in auto_masks:
                cur_labeled_mask = copy.deepcopy(label_mask[idx])
                
                # How many points can be selected for the current mask
                cnt_to_select_cur_mask = (ratio / 100.0) * all_pt_cnt * (np.sum(mask) / all_masked_px_cnt)
                cnt_to_select_cur_mask = int(cnt_to_select_cur_mask)
                if cnt_to_select_cur_mask == 0:
                    cnt_to_select_cur_mask = 1
                # cnt_to_select_cur_mask = int(round(cnt_to_select_cur_mask))

                # The index (T \ F) of unmarked points in the current mask, with a length consistent with the number of points
                in_mask_unlabel_tfmask = mask[cur_pt_img[:,0],cur_pt_img[:,1]] & (~cur_labeled_mask)

                # The entropy corresponding to the points in the current mask
                in_mask_ent = entropy_mix[in_mask_unlabel_tfmask]
                in_mask_ent_ori_idx = np.arange(all_pt_cnt)[in_mask_unlabel_tfmask]
                sorted_ent_indexes = np.argsort(in_mask_ent)
                descending_sorted_ent_indexes = sorted_ent_indexes[::-1]

                idx_to_label.extend(in_mask_ent_ori_idx[descending_sorted_ent_indexes[:cnt_to_select_cur_mask]])

            label_mask[idx][idx_to_label] = True
            if pselab_save_path:
                pseudo_label_dict = {
                    'probs_2d': np.max(pred_probs_2d,axis=1),
                    # 'ori_probs_2d':pred_probs_2d,
                    'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                    'probs_3d': np.max(pred_probs_3d,axis=1),
                    # 'ori_probs_3d':pred_probs_3d,
                    'pseudo_label_3d': pred_label_3d.astype(np.uint8),
                    'pseudo_label_ensemble':pred_label_ensemble.astype(np.uint8),
                }
                pselab_data_list.append(pseudo_label_dict)

        # pdb.set_trace()
        
        np.save(save_path,label_mask)
        mylog(f'save label mask to {save_path} !')
        if pselab_save_path:
            np.save(pselab_save_path, pselab_data_list)
            mylog(f'save pseduo label to {pselab_save_path} !')

def get_label_mask_by_feature_entropy_slic(cfg,dataset,model_2d,model_3d,ratio,save_path,logger,pselab_save_path = None):
    mylog = check_logger(logger)

    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )
    label_mask = get_cur_mask(cfg,val_loader,logger)
    # pdb.set_trace()
    k = cfg.ADA.n_neighbors
    # predictor = SamPredictor(sam_model)
    delete_road_mask = cfg.SAM.delete_road_mask
    mylog(f"batch_size for query= {cfg.ADA.batch_size},k = {cfg.ADA.n_neighbors},delete_road_mask = {cfg.SAM.delete_road_mask}")

    pselab_data_list = []
    
    with torch.no_grad():
        for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            # idx,data_batch = val_iter.__next__()
            data_batch['x'][1] = data_batch['x'][1].cuda()
            # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch['img'] = data_batch['img'].cuda()

            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch)

            pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            pred_probs_2d = pred_probs_2d.detach().cpu().numpy()

            pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)
            pred_probs_3d = pred_probs_3d.detach().cpu().numpy()

            pred_label_ensemble = (pred_probs_2d + pred_probs_3d).argmax(1)
            pred_label_2d = pred_probs_2d.argmax(1)
            pred_label_3d = pred_probs_3d.argmax(1)
            #*****************************
            # get knn mask
            #*****************************
            features_2d = preds_2d['feats'].cpu().numpy()
            features_3d = preds_3d['feats'].cpu().numpy()

            nbrs_2d = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features_2d)
            nbrs_3d = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features_3d)

            distances_2d, indices_2d = nbrs_2d.kneighbors(features_2d)
            distances_3d, indices_3d = nbrs_3d.kneighbors(features_3d)

            # neighbors_label_2d = data_batch['pseudo_label_ensemble'].numpy()[indices_2d]
            # neighbors_label_3d = data_batch['pseudo_label_ensemble'].numpy()[indices_3d]
            neighbors_label_2d = pred_label_ensemble[indices_2d]
            neighbors_label_3d = pred_label_ensemble[indices_3d]

            neighbors_cls_portion_2d = np.array(list(map(lambda row: calculate_proportions(row,cfg), neighbors_label_2d)))

            neighbors_cls_portion_3d = np.array(list(map(lambda row: calculate_proportions(row,cfg), neighbors_label_3d)))

            entropy_2d = -np.sum(neighbors_cls_portion_2d * np.log2(neighbors_cls_portion_2d + 1e-10),axis=1)
            entropy_3d = -np.sum(neighbors_cls_portion_3d * np.log2(neighbors_cls_portion_3d + 1e-10),axis=1)

            entropy_mix = entropy_2d + entropy_3d
            # pdb.set_trace()

            #*****************************
            # get sam mask
            #*****************************
            cur_pt_img = data_batch['img_indices'][0]
            image_np = data_batch['img'][0].permute(1, 2, 0).cpu().numpy()
            image_np_int8 = (image_np * 255).astype(np.uint8)

            # n_segments = 35: avg sam mask cnt
            segments_slic = slic(image_np_int8,n_segments = 35).astype(np.uint8)
            slic_masks = []
            for pixel_label in np.unique(segments_slic):
                slic_masks.append(segments_slic == np.full(segments_slic.shape,pixel_label))

            all_pt_cnt = len(cur_pt_img)
            all_masked_px_cnt = np.sum(slic_masks)
            idx_to_label = []

            for mask in slic_masks:
                cur_labeled_mask = copy.deepcopy(label_mask[idx])
                
                cnt_to_select_cur_mask = (ratio / 100.0) * all_pt_cnt * (np.sum(mask) / all_masked_px_cnt)
                cnt_to_select_cur_mask = int(cnt_to_select_cur_mask)
                if cnt_to_select_cur_mask == 0:
                    cnt_to_select_cur_mask = 1
                # cnt_to_select_cur_mask = int(round(cnt_to_select_cur_mask))

                in_mask_unlabel_tfmask = mask[cur_pt_img[:,0],cur_pt_img[:,1]] & (~cur_labeled_mask)

                in_mask_ent = entropy_mix[in_mask_unlabel_tfmask]
                in_mask_ent_ori_idx = np.arange(all_pt_cnt)[in_mask_unlabel_tfmask]
                sorted_ent_indexes = np.argsort(in_mask_ent)
                descending_sorted_ent_indexes = sorted_ent_indexes[::-1]
                idx_to_label.extend(in_mask_ent_ori_idx[descending_sorted_ent_indexes[:cnt_to_select_cur_mask]])

            label_mask[idx][idx_to_label] = True
            if pselab_save_path:
                pseudo_label_dict = {
                    'probs_2d': np.max(pred_probs_2d,axis=1),
                    # 'ori_probs_2d':pred_probs_2d,
                    'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                    'probs_3d': np.max(pred_probs_3d,axis=1),
                    # 'ori_probs_3d':pred_probs_3d,
                    'pseudo_label_3d': pred_label_3d.astype(np.uint8),
                    'pseudo_label_ensemble':pred_label_ensemble.astype(np.uint8),
                }
                pselab_data_list.append(pseudo_label_dict)

        # pdb.set_trace()
        
        np.save(save_path,label_mask)
        mylog(f'save label mask to {save_path} !')
        if pselab_save_path:
            np.save(pselab_save_path, pselab_data_list)
            mylog(f'save pseduo label to {pselab_save_path} !')
            
# Please refer to
# https://github.com/BIT-DA/Annotator

# def get_label_mask_by_annotator(cfg,dataset,model_2d,model_3d,ratio,save_path,logger,pselab_save_path = None):
#     mylog = check_logger(logger)
#     collate_fn_val = get_collate_scn(is_train = True)
#     val_loader = DataLoader(
#         dataset,
#         batch_size=1,
#         drop_last=False,
#         num_workers=cfg.DATALOADER.NUM_WORKERS,
#         worker_init_fn=worker_init_fn,
#         collate_fn=collate_fn_val
#     )
#     label_mask = get_cur_mask(cfg,val_loader,logger)
#     with torch.no_grad():
#         for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            
#             positive_points = data_batch['ori_points'][0] - data_batch['ori_points'][0].min(0)
#             positive_points = torch.from_numpy(positive_points).cuda()
#             voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(positive_points, 0.25, 100)
#             # idx,data_batch = val_iter.__next__()
#             data_batch['x'][1] = data_batch['x'][1].cuda()
#             # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
#             data_batch['img'] = data_batch['img'].cuda()

#             preds_2d = model_2d(data_batch)
#             preds_3d = model_3d(data_batch)

#             pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
#             pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)

#             # pred_label_ensemble = (pred_probs_2d + pred_probs_3d).argmax(1).cpu().numpy()
#             pred_label_ensemble = (pred_probs_2d + pred_probs_3d).argmax(1)

#             point_in_voxel_label = torch.where(point_in_voxel != -1, pred_label_ensemble[point_in_voxel], -1)

#             num_classes = cfg.MODEL_2D.NUM_CLASSES
#             class_num_count = torch.zeros((point_in_voxel_label.shape[0], num_classes + 1),
#                                             device=pred_label_ensemble.device, dtype=torch.int)

#             for i in range(1, num_classes + 1):
#                 class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
#                 class_num_count_i = torch.sum(class_num_count_i, dim=1)
#                 class_num_count[:, i] = class_num_count_i

#             class_num_probability = class_num_count / voxel_point_counts[:, None]
#             temp = torch.log2(class_num_probability)
#             log2_class_num_probability = torch.where(
#                 class_num_count != 0, temp,
#                 torch.tensor(0, device=pred_label_ensemble.device, dtype=torch.float))
#             confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

#             _, voxel_indices = torch.sort(confusion, descending=True)

#             cur_labeled_mask = copy.deepcopy(label_mask[idx]) 

#             select_point_num = (ratio/100.0) * (len(cur_labeled_mask))
#             selected_point_num = 0
#             selected_voxel_num = 0
#             selected_voxel_idx = []
#             selected_points_idx = []

#             for voxel_idx in voxel_indices:
#                 cur_points_in_voxel = point_in_voxel[voxel_idx].cpu().numpy()
#                 # If all points within the current voxel are unmarked
#                 if all(~cur_labeled_mask[cur_points_in_voxel]):
#                     selected_voxel_idx.append(voxel_idx)
#                     selected_points_idx.extend(cur_points_in_voxel[cur_points_in_voxel != -1])
#                     selected_point_num += voxel_point_counts[voxel_idx]
#                     selected_voxel_num += 1
#                     if selected_point_num > select_point_num:
#                         # mylog('selected_voxel_num',selected_voxel_num)
#                         # mylog('selected_point_num',selected_point_num)
#                         # mylog('len(selected_points_idx)',len(selected_points_idx))
#                         break
#             label_mask[idx][selected_points_idx] = True
            
#         # pdb.set_trace()
        
#         np.save(save_path,label_mask)
#         mylog(f'save label mask to {save_path} !')


# Please refer to
# https://github.com/tsunghan-wu/ReDAL

# def get_label_mask_by_redal(cfg,dataset,model_2d,model_3d,ratio,save_path,logger,pselab_save_path = None):
#     import warnings
#     warnings.simplefilter(action='ignore', category=FutureWarning)
    
#     mylog = check_logger(logger)
#     collate_fn_val = get_collate_scn(is_train = True)
#     val_loader = DataLoader(
#         dataset,
#         batch_size=1,
#         drop_last=False,
#         num_workers=cfg.DATALOADER.NUM_WORKERS,
#         worker_init_fn=worker_init_fn,
#         collate_fn=collate_fn_val
#     )
#     label_mask = get_cur_mask(cfg,val_loader,logger)
#     curvature = []
#     curvature.extend(np.load(cfg.ADA.ReDAL.curvature_path, allow_pickle=True))
#     supervoxel_idx = []
#     supervoxel_idx.extend(np.load(cfg.ADA.ReDAL.supervoxel_path, allow_pickle=True))
#     alpha = cfg.ADA.ReDAL.alpha
#     beta = cfg.ADA.ReDAL.beta
#     gamma = cfg.ADA.ReDAL.gamma

#     num_clusters = cfg.ADA.ReDAL.num_clusters
#     decay_rate = cfg.ADA.ReDAL.decay_rate
    
#     with torch.no_grad():
#         for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
#             cur_supvox_id = supervoxel_idx[idx]
#             assert(len(cur_supvox_id) == len(label_mask[idx])),"len(cur_supvox_id) == len(label_mask[idx]) must be satisfied."
            
#             data_batch['x'][1] = data_batch['x'][1].cuda()
#             # data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
#             data_batch['img'] = data_batch['img'].cuda()

#             preds_2d = model_2d(data_batch)
#             preds_3d = model_3d(data_batch)

#             pred_probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
#             pred_probs_3d = F.softmax(preds_3d['seg_logit'], dim=1)

#             pred_probs_ensemble = (pred_probs_2d + pred_probs_3d)*0.5

#             # [Uncertainty Calculation]
#             # 1. Take out the segmentation prediction of all points beloning to a single point cloud.
#             # 2. Then, perform uncertainty calculation.
#             uncertain = torch.mean(-pred_probs_ensemble * torch.log2(pred_probs_ensemble + 1e-12), dim=1)
#             uncertain = uncertain.cpu().detach().numpy()

#             # [Region Feature Extraction]
#             # 1. Take out the supervoxel ID of all points belonging to a single point cloud.
#             # 2. Gathering region features using torch_scatter library, which is convenient
#             #    for us to gather all point features belonging to the same supervoxel ID.
#             # 3. Finally, we combine them into a numpy array (named as all_feats) for subsequent usage.
            
#             features_2d = preds_2d['feats']
#             features_3d = preds_3d['feats']
#             features_mean_2d = scatter_mean(features_2d, torch.tensor(cur_supvox_id).cuda(), dim=0).cpu().numpy()
#             non_zero_mask_2d = np.any(features_mean_2d != 0, axis=1)
#             features_mean_2d = features_mean_2d[non_zero_mask_2d]

#             features_mean_3d = scatter_mean(features_3d, torch.tensor(cur_supvox_id).cuda(), dim=0).cpu().numpy()
#             non_zero_mask_3d = np.any(features_mean_3d != 0, axis=1)
#             features_mean_3d = features_mean_3d[non_zero_mask_3d]
            
#             # Per-point information score. (Refer to the equation 4 in the main paper.)
#             cur_curvature = curvature[idx]
#             colorgrad = np.zeros_like(cur_curvature)
#             point_score = alpha * uncertain + beta * colorgrad + gamma * cur_curvature

#             # We gather the region information scores using "pandas groupby method" as shown below.
#             # This is efficient to gather per-point scores belonging to the same supervoxel ID into one.
#             df = pd.DataFrame({'id': cur_supvox_id, 'val': point_score})
#             df1 = df.groupby('id')['val'].agg(['count', 'mean']).reset_index()

#             # delete mean score of labeled regions
#             labeled_region_idxs = np.unique(np.array(cur_supvox_id)[label_mask[idx]])
#             if len(labeled_region_idxs) != 0:
#                 unlabel_spv_idxs = ~df1['id'].isin(labeled_region_idxs)
#                 df2 = df1[unlabel_spv_idxs]
#                 df2.reset_index(drop=True,inplace=True)
#                 features_mean_2d = features_mean_2d[unlabel_spv_idxs]
#                 features_mean_3d = features_mean_3d[unlabel_spv_idxs]
#             else:
#                 df2 = df1.copy()
#             # pdb.set_trace()
#             # [Diversity-aware Selection]
#             # The "importance_reweight" function is the implementation of Sec.3.3.2 in the main paper.
#             # clustering

#             def importance_reweight(scores, features):
#                 # sorted (first time)
#                 # sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
#                 # features = features[sorted_idx]
#                 # selected_samples = sorted(scores, reverse=True)

#                 # trim_region = cfg.ADA.ReDAL.trim_region
#                 # trim_rate = cfg.ADA.ReDAL.trim_rate

#                 # if trim_region is True:
#                 #     N = features.shape[0] * trim_rate
#                 #     features = features[:N]
#                 #     selected_samples = selected_samples[:N]

#                 # Prevent errors when the number of features is less than the number of clusters
#                 n_clusters = num_clusters
#                 feat_cnt = len(features)
#                 if feat_cnt < num_clusters:
#                     n_clusters = feat_cnt
#                 kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#                 kmeans.fit(features)

#                 clusters = kmeans.labels_
#                 # importance re-weighting
#                 N = features.shape[0]
#                 importance_arr = [1.0 for _ in range(num_clusters)]
#                 for i in range(N):
#                     cluster_i = clusters[i]
#                     cluster_importance = importance_arr[cluster_i]
#                     scores.loc[i,'mean'] *= cluster_importance
#                     importance_arr[cluster_i] *= decay_rate
#                 # sorted (second time)
#                 # pdb.set_trace()
#                 return scores
#                 # selected_samples = sorted(scores, reverse=True)
#                 # return selected_samples
#             # pdb.set_trace()
#             reweighted_scores_2d = importance_reweight(df2.copy(),features_mean_2d)
#             reweighted_scores_3d = importance_reweight(df2.copy(),features_mean_3d)

#             merged_df = pd.merge(reweighted_scores_2d, reweighted_scores_3d, on='id', suffixes=('_df1', '_df2'))

#             merged_df['mean'] = merged_df['mean_df1'] + merged_df['mean_df2']
#             merged_df = merged_df[['mean', 'count_df1', 'id']].rename(columns={'count_df1': 'count'})
#             # reweighted_scores_ensemble = reweighted_scores_2d + reweighted_scores_3d
#             reweighted_scores_ensemble_sorted = merged_df.sort_values(by='mean', ascending=False)
            
#             # select regions
#             selected_spv_idx = []
#             point_num_to_select = int((ratio/100.0) * (len(label_mask[idx])))
#             selected_cnt = 0
#             for index, row in reweighted_scores_ensemble_sorted.iterrows():
#                 selected_spv_idx.append(row['id'])
#                 label_mask[idx][cur_supvox_id == row['id']] = True
#                 selected_cnt += row['count']
#                 if selected_cnt >= point_num_to_select:
#                     break

#         np.save(save_path,label_mask)
#         mylog(f'save label mask to {save_path} !')