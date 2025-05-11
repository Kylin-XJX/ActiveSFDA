import pdb
import pickle
import warnings
from pclpy import pcl
from tqdm import tqdm
import numpy as np
import os

def SupervoxelClustering_nuScenes():
    VCCS_param = {
        'voxel_resolution' : 0.5,  
        'seed_resolution' : 5.0,   
        'color_importance' : 0.0,  
        'spatial_importance' : 1.0,
        'normal_importance' : 0.0,  
    }
    pkl_path = ''
    save_path = '/train_singapore_supervoxel_{}_{}.npy'.format(
        VCCS_param['voxel_resolution'],
        VCCS_param['seed_resolution']
    )

    print('pkl_path:',pkl_path)
    print('save_path:',save_path)
    print('VCCS_param:\n',VCCS_param)
    data = []
    super_voxel_labels = []

    print("Loading pkl...")
    with open(pkl_path, 'rb') as f:
        data.extend(pickle.load(f))

    for data_item in tqdm(data):

        points = data_item['points']
        rgba = np.full((len(points),4),0)
        cloud = pcl.PointCloud.PointXYZRGBA()
        cloud = cloud.from_array(points,rgba)
        
        # Create a supervoxel object
        supervoxel = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(VCCS_param['voxel_resolution'], VCCS_param['seed_resolution'])
        supervoxel.setInputCloud(cloud)
        supervoxel.setColorImportance(VCCS_param['color_importance'])
        supervoxel.setSpatialImportance(VCCS_param['spatial_importance'])
        supervoxel.setNormalImportance(VCCS_param['normal_importance'])

        # Perform supervoxel segmentation
        supervoxel_clusters = pcl.vectors.map_uint32t_PointXYZRGBA()
        supervoxel.extract(supervoxel_clusters)

        # Obtain voxel labels after clustering
        labeled_cloud = supervoxel.getLabeledCloud()
        labels = []
        for point in labeled_cloud.points:
            labels.append(point.label)
        
        super_voxel_labels.append(labels)
    
    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        warnings.warn('Make a new directory: {}'.format(save_dir))
        os.makedirs(save_dir,exist_ok=True)

    np.save(save_path,super_voxel_labels)
    print(f'save label mask to {save_path} !')


def SupervoxelClustering_semkitti():
    VCCS_param = {
        'voxel_resolution' : 0.5,  
        'seed_resolution' : 10.0,  
        'color_importance' : 0.0,  
        'spatial_importance' : 1.0,
        'normal_importance' : 0.0,  
    }
    points_path = ''
    save_path = '/train_semkitti_supervoxel_{}_{}.npy'.format(
        VCCS_param['voxel_resolution'],
        VCCS_param['seed_resolution']
    )

    print('points_path:',points_path)
    print('save_path:',save_path)
    print('VCCS_param:\n',VCCS_param)

    points_data = []
    points_data.extend(np.load(points_path, allow_pickle=True))

    super_voxel_labels = []

    for points in tqdm(points_data):
        # pdb.set_trace()
        rgba = np.full((len(points),4),0)
        cloud = pcl.PointCloud.PointXYZRGBA()
        cloud = cloud.from_array(points,rgba)
        
        # Create a supervoxel object
        supervoxel = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(VCCS_param['voxel_resolution'], VCCS_param['seed_resolution'])
        supervoxel.setInputCloud(cloud)
        supervoxel.setColorImportance(VCCS_param['color_importance'])
        supervoxel.setSpatialImportance(VCCS_param['spatial_importance'])
        supervoxel.setNormalImportance(VCCS_param['normal_importance'])

        # Perform supervoxel segmentation
        supervoxel_clusters = pcl.vectors.map_uint32t_PointXYZRGBA()
        supervoxel.extract(supervoxel_clusters)

        # Obtain voxel labels after clustering
        labeled_cloud = supervoxel.getLabeledCloud()
        labels = []
        for point in labeled_cloud.points:
            labels.append(point.label)
        
        super_voxel_labels.append(labels)
    
    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        warnings.warn('Make a new directory: {}'.format(save_dir))
        os.makedirs(save_dir,exist_ok=True)

    np.save(save_path,super_voxel_labels)
    print(f'save label mask to {save_path} !')

def main():
    # SupervoxelClustering_nuScenes()
    SupervoxelClustering_semkitti()

if __name__ == '__main__':
    main()