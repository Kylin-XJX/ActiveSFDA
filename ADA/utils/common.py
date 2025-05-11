from tqdm import tqdm
from xmuda.common.config.base import CN
import torch.nn.functional as F
import numpy as np
import torch
from utils.utils import get_mask_prompt,mask_to_boxes
from scipy.ndimage import morphology

def add_config(cfg,dataset_target_type = 'NuScenesLidarSegSCN'):
    cfg.OPTIMIZER.FACTOR_2D = 1.0
    cfg.OPTIMIZER.FACTOR_3D = 1.0

    cfg.DATASET_TARGET[dataset_target_type].label_mask_path = ''
    cfg.DATASET_TARGET[dataset_target_type].return_ori_points = False
    cfg.DATASET_TARGET[dataset_target_type].new_pselab = False
    # cfg.DATASET_TARGET[dataset_target_type].no_refine = False

    cfg.TRAIN.XMUDA.lambda_ce_trg = 1.0

    cfg.ADA = CN()
    cfg.ADA.query_iters = []
    cfg.ADA.budget = 0.0
    cfg.ADA.save_dir = ''
    cfg.ADA.query_function_name = ''
    cfg.ADA.batch_size = 1
    cfg.ADA.n_neighbors = 4
    cfg.ADA.load_mask_from_disk = False
    cfg.ADA.update_pselab = True
    cfg.ADA.pselab_save_dir = ''
    cfg.ADA.r_threshold = 0.85
    
    cfg.ADA.ReDAL = CN()
    cfg.ADA.ReDAL.curvature_path = ""
    cfg.ADA.ReDAL.supervoxel_path = ""
    cfg.ADA.ReDAL.alpha = 1.0
    cfg.ADA.ReDAL.beta = 0.0
    cfg.ADA.ReDAL.gamma = 0.05
    cfg.ADA.ReDAL.num_clusters = 20
    cfg.ADA.ReDAL.decay_rate = 0.95

    cfg.SAM = CN()
    cfg.SAM.CKPT = ""
    cfg.SAM.MODEL_TYPE = ""
    cfg.SAM.IMG_SIZE = (225,400)
    cfg.SAM.delete_road_mask = False


def get_sam_masks(cfg,predictor,
                  pred_probs_2d,pred_probs_3d,
                  data_batch,delete_road_mask = False):

    image_np = data_batch['img'][0].permute(1, 2, 0).cpu().numpy()
    # Convert the image to [0-255], int8
    image_np_int8 = (image_np * 255).astype(np.uint8)
    cur_pt_img = data_batch['img_indices'][0]
    mask_prompts = get_mask_prompt(
        pred_probs_2d,
        pred_probs_3d,
        cur_pt_img,
        cfg.MODEL_2D.NUM_CLASSES,
        cfg.SAM.IMG_SIZE
    )
    all_boxes = mask_to_boxes(mask_prompts)

    all_boxes_collated = []
    prompt_cls = []
    for cls_idx in range(cfg.MODEL_2D.NUM_CLASSES):
        if len(all_boxes[cls_idx]) == 0:
                continue
        for box in all_boxes[cls_idx]:
            all_boxes_collated.append(box)
            prompt_cls.append(cls_idx)
    prompt_cls = np.array(prompt_cls)
    
    predictor.set_image(image_np_int8)
    transformed_boxes = predictor.transform.apply_boxes_torch(torch.tensor(all_boxes_collated).to('cuda'), image_np_int8.shape[:2])
    masks,iou_predictions, low_res_masks = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes=transformed_boxes,
        mask_input=None,
        multimask_output = False,
        return_logits=False
    )

    masks_np = masks.cpu().numpy()
    area = [np.sum(mask) for mask in masks_np]
    # Remove the mask with the largest area (road, ground)
    if delete_road_mask:
        masks_np = np.delete(masks_np,np.argmax(area),axis=0)

    dilation_kernel_size = (3, 3)  
    dilation_iterations = 1 

    dilated_masks = []
    for mask in masks_np: 
        # Perform dilation operation on binary images
        dilated_mask = morphology.binary_dilation(mask[0], structure=np.ones(dilation_kernel_size), iterations=dilation_iterations)
        dilated_masks.append(dilated_mask)

    return dilated_masks


def get_cur_mask(cfg,val_loader,logger,debug = False):
    dataset_target_type = cfg.DATASET_TARGET.TYPE
    cur_mask_path = cfg.DATASET_TARGET[dataset_target_type].label_mask_path
    if cur_mask_path:
        if logger:
            logger.info(f'loading label mask from {cur_mask_path}')
        else:
            print(f'loading label mask from {cur_mask_path}')
        label_mask = []
        label_mask.extend(np.load(cur_mask_path, allow_pickle=True))
    else:
        label_mask = []
        if logger:
            logger.info('init label mask...')
        else:  
            print('init label mask...')
        for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            mask_len = len(data['seg_label'])
            if debug:
                if idx<10:
                    print(mask_len)
                else:
                    break
            mask = np.zeros(mask_len,dtype=bool)
            label_mask.append(mask)
    return np.array(label_mask,dtype=object)

def calculate_proportions(row,cfg):
    # Use numpy's bincount function to calculate the frequency of each element
    # Because the range of the element is [0,5], minlength is set to 6
    counts = np.bincount(row, minlength=cfg.MODEL_2D.NUM_CLASSES) 
    # Calculate the proportion of each element
    proportions = counts / len(row)
    return proportions

def check_logger(logger):
    if logger:
        mylog = logger.info
    else:
        mylog = print
    return mylog