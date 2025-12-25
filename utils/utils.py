import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.ndimage import morphology
import cv2
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from SAM.segment_anything import sam_model_registry, SamPredictor
from matplotlib.colors import ListedColormap


def densify_mask(sparse_mask):
    # the size of the kernel and the number of iterations
    dilation_kernel_size = (9, 9)  
    dilation_iterations = 1

    # dilation operation on binary images
    dilated_image = morphology.binary_dilation(sparse_mask, structure=np.ones(dilation_kernel_size), iterations=dilation_iterations)

    # erosion
    erosion_kernel_size = (3, 3)
    erosion_iterations = 1

    # erosion operation on the dilation image
    eroded_image = morphology.binary_erosion(dilated_image, structure=np.ones(erosion_kernel_size), iterations=erosion_iterations)
 
    # Blur the dilation image to eliminate jagged edges caused by sparse point cloud
    gray_image = (eroded_image * 255).astype(np.uint8)
    # Convert grayscale images to color images using OpenCV
    gray_image_opencv = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    # the kernel size for Gaussian blur
    kernel_size = (19,19) 
    # Gaussian blur on binary images
    blurred_image = cv2.GaussianBlur(gray_image_opencv, kernel_size, 0)

    blurred_gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    # threshold
    threshold_value = 128
    # Thresholding operation on grayscale images
    dense_mask = (blurred_gray_image > threshold_value)
    return dense_mask

def get_SAM_Predictor(
        sam_checkpoint = "" ,
        model_type = "vit_h",
        device = "cuda"
        ):
    # sam_checkpoint = ""
    # model_type = "vit_h"
    # device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def get_mask_prompt(pred_prob_2d,pred_prob_3d,point_img,NUM_CLASSES,img_size):
    """
    Return the mask for each class based on the predicted probability of 2D3D
    Even if there are no such objects, a mask will still be returned, and the mask will be empty at this time
    Args:
        pred_prob_2d (np.array): -
        pred_prob_3d (np.array): -
        point_img (np.array): -
        NUM_CLASSES (int): -
        img_size (tuple): -

    Returns:
        np.array: mask of each class, even if there are no pixels in this type of object, will return an empty mask
    """
    mixed_prob = 0.5*(pred_prob_2d+pred_prob_3d)
    mixed_predictions = np.argmax(mixed_prob, axis=1)
    max_mixed_prob = np.max(mixed_prob, axis=1)

    mask_prompts = []
    for pt_class in range(NUM_CLASSES):
        mask = np.zeros(img_size,dtype=bool)
        # Convert the array type to Boolean and set it to False
        curr_idx = mixed_predictions == pt_class
        curr_probs = max_mixed_prob[curr_idx]
        percentile_thresh = 0
        mask_coords = point_img[curr_idx][curr_probs > percentile_thresh]
        mask[mask_coords[:,0],mask_coords[:,1]] = True
        mask = densify_mask(mask)
        mask_prompts.append(mask)

    return mask_prompts

def combine_masks(masks):
    final_mask = np.zeros(masks.shape[-3:],dtype=bool)
    for mask in masks:
        final_mask = np.logical_or(final_mask,mask)
    final_mask = final_mask[0,:,:]
    # Return shape as [H, W]
    return final_mask

def combine_masks_logits(masks):
    final_mask = np.zeros(masks.shape[-2:])
    for mask in masks:
        final_mask = final_mask+mask
    final_mask = final_mask.squeeze(0)
    return final_mask

def mask_to_boxes(mask_prompts):
    all_boxes = []
    for mask in mask_prompts:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y,x + w, y + h])
        all_boxes.append(boxes)
    return all_boxes