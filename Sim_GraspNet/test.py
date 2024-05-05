import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
#   print(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(ROOT_DIR)
from collision_detector import ModelFreeCollisionDetector
from models.SimGraspDataset import simgraspdata
from models.SimGraspNet import Sim_Grasp_Net, pred_decode
from torch.utils.data import DataLoader
import open3d as o3d
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from loss import get_loss
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default="/media/juncheng/Disk4T1/Sim-Grasp/Sim_GraspNet/logs/log/sim_grasp_epoch59.tar")
#parser.add_argument('--checkpoint_path', help='Model checkpoint path', default="/home/juncheng/Downloads/sim_grasp_epoch04.tar")

parser.add_argument('--dump_dir', help='Dump dir to save outputs', default="/media/juncheng/Disk4T1/Sim-Grasp/Sim_GraspNet/logs/log/")
parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.5, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=True)
parser.add_argument('--eval', action='store_true', default=False)
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)
#def translation_distance_matrix(translations):
    #return torch.norm(translations[:, None, :] - translations[None, :, :], dim=-1)
def translation_distance_matrix(translations1, translations2):
    return torch.norm(translations1[:, None, :] - translations2[None, :, :], dim=-1)
#def rotation_distance_matrix(rotations):
    #return torch.acos((torch.einsum('bij,bkj->bik', rotations, rotations.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
def rotation_distance_matrix(rotations1, rotations2):
    return torch.acos((torch.einsum('bij,bkj->bik', rotations1, rotations2.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)

def grasp_pose_nms_vectorized2(grasps, thd=100, tha=np.deg2rad(10), K=100):
    translations = torch.tensor(grasps[:, :3])
    rotations = torch.tensor(grasps[:, 3:12]).view(-1, 3, 3)
    confidences = torch.tensor(grasps[:, 12])

    t_dist = translation_distance_matrix(translations)
    r_dist = rotation_distance_matrix(rotations)

    # Create a mask for pairs of grasps that are too close
    too_close_mask = (t_dist < thd) & (r_dist < tha)

    # Keep a grasp if it is not too close to any higher-confidence grasp
    keep_mask = ~(too_close_mask.triu(diagonal=1).any(dim=1))

    # Sort grasps by confidence and keep the top K
    sorted_indices = torch.argsort(confidences, descending=True)
    top_k_indices = sorted_indices[:K]
    keep_indices = top_k_indices[keep_mask[sorted_indices][:K]]

    return grasps[keep_indices]

def grasp_pose_nms_vectorized(grasps, thd=0.5, tha=np.deg2rad(30), K=100):
    translations = torch.tensor(grasps[:, :3])
    rotations = torch.tensor(grasps[:, 3:12]).view(-1, 3, 3)
    confidences = torch.tensor(grasps[:, 12])

    # Sort grasps by confidence in descending order
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_translations = translations[sorted_indices]
    sorted_rotations = rotations[sorted_indices]
    sorted_confidences = confidences[sorted_indices]

    keep_indices = []
    while len(keep_indices) < K and sorted_translations.shape[0] > 0:
        # Always keep the grasp with the highest confidence
        keep_indices.append(sorted_indices[0])
        current_translation = sorted_translations[0:1]
        current_rotation = sorted_rotations[0:1]

        # Compute distances to the remaining grasps
        t_dist = translation_distance_matrix(current_translation, sorted_translations[1:])
        r_dist = rotation_distance_matrix(current_rotation, sorted_rotations[1:])

        # Find grasps that are not too close
        not_too_close = (t_dist >= thd) & (r_dist >= tha)

        # Keep only the grasps that are not too close
        sorted_translations = sorted_translations[1:][not_too_close.squeeze()]
        sorted_rotations = sorted_rotations[1:][not_too_close.squeeze()]
        sorted_indices = sorted_indices[1:][not_too_close.squeeze()]

    return grasps[torch.tensor(keep_indices)]
# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

def get_net():
    # Init the model
    net =  Sim_Grasp_Net(seed_feat_dim=256,is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def get_and_process_data_sim(data_dir,segmentation_path):
    # load data
    pcd = np.load(data_dir, allow_pickle=True)['arr_0']

    #segmentation = np.load(segmentation_path, allow_pickle=True)['arr_0']

    #point_idxs = np.where(segmentation != 1)[0]

    # Convert to Open3D point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcd[:, 0:3].astype(np.float32))
    cloud.normals = o3d.utility.Vector3dVector(pcd[:, 3:6].astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(pcd[:, 6:9].astype(np.float32))

    # Downsample the point cloud to 20,000 points
    #downsampled_cloud = cloud.uniform_down_sample(int(len(cloud.points) / 40000))
    downsampled_cloud=cloud
    # Convert back to NumPy array
    downsampled_pcd = np.asarray(downsampled_cloud.points)


    
   
    #pc,centroid,m=pc_normalize(pcd[:,0:3])
    pc,centroid,m=pc_normalize(np.asarray(downsampled_cloud.points))
    N = pc.shape[0]  # Number of points in the point cloud

    # Initialize cloud_sampled with shape (N, 6)
    cloud_sampled = np.zeros((N, 6))

    cloud_sampled[:,0:3] = pc
    #cloud_sampled[:,3:6]=pcd[:,3:6]
    cloud_sampled[:,3:6]=np.asarray(downsampled_cloud.normals)
    #color_sampled = pcd[:,6:9]
    #print(cloud_sampled,color_sampled)
    #cloud_sampled=cloud_sampled[point_idxs]
    #color_sampled=color_sampled[point_idxs]
    end_points = dict()
    #cloud = o3d.geometry.PointCloud()
    #cloud.points = o3d.utility.Vector3dVector((pcd[:,0:3]).astype(np.float32))
    #cloud.colors = o3d.utility.Vector3dVector(pcd[:,6:9].astype(np.float32))
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['centroid']=centroid
    end_points['m']=m
    #end_points['cloud_colors'] = color_sampled
   
    return end_points, cloud

def get_grasps(net, end_points,clouds):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg,gg_array

def collision_detection(gg, cloud,gg_array):
    mfcdetector = ModelFreeCollisionDetector(cloud, gg_array,voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect2(gg)
    #gg=mfcdetector.detect2(gg)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    #gg.nms()
    gg.sort_by_score()
    gg = gg[:550]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir,segmentation_path):
    net = get_net()
    end_points, cloud = get_and_process_data_sim(data_dir,segmentation_path)
    gg,gg_array = get_grasps(net, end_points,cloud)
    #if cfgs.collision_thresh > 0:
        #print("here")
    gg = collision_detection(gg, np.array(cloud.points),gg_array)
    vis_grasps(gg, cloud)


if __name__=='__main__':
    #data_dir = '/media/juncheng/Disk4T1/Sim-Grasp/pointcloud_train_grasp/5.npz'
    data_dir ='/media/juncheng/ubuntu_data1/Sim-Suction-API/test_similar_pointcloud/29_31.npz'
    #data_dir = '/media/juncheng/Disk4T1/Sim-Grasp/single_pointcloud/15_12.npz'
    #data_dir = '/media/juncheng/Disk4T1/Sim-Grasp/pointcloud_train/32.npz'

    #data_dir ='/media/juncheng/ubuntu_data1/Sim-Suction-API/sim_suction_policy/demo/demo.npz'
    segmentation_path = '/media/juncheng/Disk4T1/Sim-Grasp/pointcloud_train_grasp/15.npz'

    demo(data_dir,segmentation_path)
