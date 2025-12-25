import torch
from functools import partial


def collate_scn_base(input_dict_list, output_orig, output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    labels=[]

    if output_image:
        imgs = []
        img_idxs = []

    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
    
    return_probs = 'ori_probs_2d' in input_dict_list[0].keys()
    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    output_label_mask = 'label_mask' in input_dict_list[0].keys()
    return_ori_points = 'ori_points' in input_dict_list[0].keys()

    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
        pseudo_label_ensemble = []
        # ori_pseudo_label_2d = []
        # ori_pseudo_label_3d = []
        if return_probs:
            ori_probs_2d = []
            ori_probs_3d = []
    
    if output_label_mask:
        label_mask = []
    if return_ori_points:
        ori_points = []

    for idx, input_dict in enumerate(input_dict_list):
        coords = torch.from_numpy(input_dict['coords'])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict['feats']))
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))

        if output_label_mask:
            label_mask.append(torch.from_numpy(input_dict['label_mask']))

        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])
        if output_orig:
            orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
        if return_ori_points:
            ori_points.append(input_dict['ori_points'])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            # ori_pseudo_label_2d.append(torch.from_numpy(input_dict['ori_pseudo_label_2d']))

            if return_probs:
                ori_probs_2d.append(torch.from_numpy(input_dict['ori_probs_2d']))

            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
                # ori_pseudo_label_3d.append(torch.from_numpy(input_dict['ori_pseudo_label_3d']))
                if return_probs:
                    ori_probs_3d.append(torch.from_numpy(input_dict['ori_probs_3d']))
            if input_dict['pseudo_label_ensemble'] is not None:
                pseudo_label_ensemble.append(torch.from_numpy(input_dict['pseudo_label_ensemble']))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
        
    if output_label_mask:
        label_mask = torch.cat(label_mask, 0)
        out_dict['label_mask'] = label_mask

    # if output_image:
    #     out_dict['img'] = torch.stack(imgs)
    #     out_dict['img_indices'] = img_idxs
        
    if output_image:
        try:
            out_dict['img'] = torch.stack(imgs)
        except RuntimeError:
            max_H = max([img.shape[1] for img in imgs])
            max_W = max([img.shape[2] for img in imgs])
            out_dict['img'] = torch.zeros((len(imgs), 3, max_H, max_W))
            for j in range(len(imgs)):
                out_dict['img'][j, :, :imgs[j].shape[1], :imgs[j].shape[2]] = imgs[j]
        out_dict['img_indices'] = img_idxs
    if output_orig:
        out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if return_ori_points:
        out_dict['ori_points'] = ori_points
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        # out_dict['ori_pseudo_label_2d'] = torch.cat(ori_pseudo_label_2d, 0)
        if return_probs:
            out_dict['ori_probs_2d'] = torch.cat(ori_probs_2d, 0)

        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
        # out_dict['ori_pseudo_label_3d'] = torch.cat(ori_pseudo_label_3d, 0) if ori_pseudo_label_3d else ori_pseudo_label_3d
        out_dict['pseudo_label_ensemble'] = torch.cat(pseudo_label_ensemble, 0) if pseudo_label_ensemble else pseudo_label_ensemble
        if return_probs:
            out_dict['ori_probs_3d'] = torch.cat(ori_probs_3d, 0) 

    return out_dict


def get_collate_scn(is_train):
    return partial(collate_scn_base,
                   output_orig=not is_train,
                   )
