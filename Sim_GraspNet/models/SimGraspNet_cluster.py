
import os
import sys
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from models.SimGraspDataset import simgraspdata
from models.modules import ApproachVecNet, GraspAffordanceNet,GroupNet, PoseNet
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from torch.utils.data import  DataLoader
from sim_suction_model.utils.pointnet2_model import Pointnet2_scorenet
from loss_utils import generate_grasp_views,batch_viewpoint_params_to_matrix,batch_rot_matrix,batch_viewpoint_params_to_matrix_data
from knn.knn_modules import knn
GRASPNESS_THRESHOLD = 0.001
NUM_VIEW = 800
M_POINT = 2048
import open3d as o3d    
import matplotlib
from backbone import Pointnet2Backbone
import copy

import torch.nn.functional as F

def visualize_vectors_with_scores(unique_points, view_scores_matrix, views, fixed_length=0.1):
    """
    Visualizes unique points with vectors indicating significant views.
    Vectors have a fixed length and are colored from red (low score) to green (high score).

    Parameters:
    - unique_points: Nx3 numpy array of unique points' coordinates.
    - view_scores_matrix: Nx300 matrix with scores for each view of each unique point.
    - views: 300x3 array of view direction vectors from Fibonacci lattice.
    - fixed_length: Fixed length for each vector.
    """
    # Initialize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)

    # Initialize lists for lines and colors
    lines = []
    colors = []

    # Determine score range for color mapping
    min_score = np.min(view_scores_matrix[view_scores_matrix > 0])
    max_score = np.max(view_scores_matrix)

    # Function to map scores to colors
    def score_to_color(score, min_score, max_score):
        normalized_score = (score - min_score) / (max_score - min_score)
        return [1-normalized_score, normalized_score, 0]  # RGB color

    # Process each unique point and its scores
    N = len(unique_points)
    for i in range(N):
        for view_idx, score in enumerate(view_scores_matrix[i]):
            if score > 0:
                direction = views[view_idx]
                direction_normalized = direction / np.linalg.norm(direction)
                end_point = unique_points[i] + direction_normalized * fixed_length
                lines.append([unique_points[i], end_point])
                colors.append(score_to_color(score, min_score, max_score))

    # Preparing line set for visualization
    lines_idx = [[i, i+1] for i in range(0, len(lines)*2, 2)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack(lines)),
        lines=o3d.utility.Vector2iVector(lines_idx)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd, line_set])
def visualize_with_score_mask(segmented_point_cloud, dense_scores, rgb_colors=None):
    # Normalize the scores for colormap
    normalized_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))
    
    # Get color map from Matplotlib
    cmap = matplotlib.cm.get_cmap('plasma')
    score_colors = cmap(normalized_scores)[:, :3]  # RGB colors based on scores
    
    # Set alpha for blending
    alpha = 0.7
    
    # If rgb_colors is not provided, use uniform gray for all points
    if rgb_colors is None:
        rgb_colors = np.full_like(segmented_point_cloud, fill_value=127)  # Gray
    
    # Blend the score colors with the rgb_colors
    blended_colors = alpha * score_colors + (1 - alpha) * rgb_colors / 255.0
    
    # Create a point cloud object for the dense points
    dense_pcd = o3d.geometry.PointCloud()
    dense_pcd.points = o3d.utility.Vector3dVector(segmented_point_cloud)
    dense_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Uniform gray color
    
    # Create another point cloud object for the blended colors (score mask)
    score_mask_pcd = o3d.geometry.PointCloud()
    score_mask_pcd.points = o3d.utility.Vector3dVector(segmented_point_cloud)
    score_mask_pcd.colors = o3d.utility.Vector3dVector(blended_colors)
    
    # Display the dense points first, then overlay the score mask
    #o3d.visualization.draw_geometries([dense_pcd], window_name="Dense Point Cloud")
    o3d.visualization.draw_geometries([score_mask_pcd], window_name="Score Mask Overlay")

    return dense_pcd, score_mask_pcd
def fibonacci_sphere(samples=300):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

class Sim_Grasp_Net(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=256, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.M_points = M_POINT
        self.num_view = NUM_VIEW
        self.num_angle=12
        self.num_depth=7
        self.backbone  = Pointnet2_scorenet(input_chann=6, k_score=1)


        self.graspable = GraspAffordanceNet(seed_feature_dim=self.seed_feature_dim)
        self.rotation = ApproachVecNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.group = GroupNet(nsample=64, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.posead = PoseNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        # point-wise features
        pointcloud = end_points['point_clouds']
        B, point_num, _ = pointcloud.shape
        pointcloud = pointcloud.permute(0, 2, 1)
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)

        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim

        graspness_score = end_points['graspness_score'].squeeze(2)

        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = graspness_mask
        seed_features_graspable = []
        seed_xyz_graspable = []

        graspable_num_batch = 0.
        seed_index= []
        seed_xyz=seed_xyz.permute(0, 2, 1)

        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3
            #print(cur_seed_xyz.shape)
            if cur_seed_xyz.shape[0] == 0:
                # Randomly select a point and its corresponding features
                random_idx = torch.randint(0, seed_xyz[i].shape[0], (1,))
                cur_seed_xyz = seed_xyz[i][random_idx]
                cur_feat = seed_features_flipped[i][random_idx]
            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*Ns

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)


        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B

        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat


        if self.is_training:
            dense_points = end_points['coors']  # (B, 20000, 3)
            batch_size, dense_N, _ = dense_points.size()

            unique_t_ori_points = end_points['sparse_points']  # (B, N, 3)
            normalized_scores = end_points['normalized_scores']  # (B, N)
            approach_directions = end_points['rotated_approach_directions']  # (B, N, 3, 3)
            normalized_view_score= end_points['normalized_view_score']#(B,N,3)
            normalized_grasp_score=end_points['normalized_grasp_score'] # (B,N, 3, 12, 7)

            template_views = generate_grasp_views(NUM_VIEW).to(end_points['normalized_view_score'].device)  # (num_view, 3)            print(template_views)
            #batch_view_scores = torch.zeros((batch_size, num_points, num_view), device=normalized_view_score.device)
            
            end_points_view_score_list = []
            batch_grasp_point_list = []
            end_points_grasp_score_list = []
            end_points_view_rot_list = []
            pointwise_label_list=[]

            end_points_view_score = torch.zeros(batch_size, self.M_points, NUM_VIEW, device=seed_xyz_graspable.device)
            batch_grasp_point_tensor = torch.zeros(batch_size, self.M_points, 3, device=seed_xyz_graspable.device)
            end_points_grasp_score = torch.zeros(batch_size, self.M_points, NUM_VIEW, 12, 7, device=seed_xyz_graspable.device)
            end_points_view_rot = torch.zeros(batch_size, self.M_points, NUM_VIEW,3, 3, device=seed_xyz_graspable.device)
            end_points_affordance=torch.zeros(batch_size, dense_N, device=seed_xyz_graspable.device)
            for i in range(batch_size):

                valid_mask = torch.any(unique_t_ori_points[i] != 0, dim=1)
                #print(unique_t_ori_points.shape)
                valid_sparse_points = unique_t_ori_points[i][valid_mask] # (B, N, 3)
                valid_normalized_scores = normalized_scores[i][valid_mask]# (B, N)
                valid_approach_directions = approach_directions[i][valid_mask] # (B, N, 3, 3)
                valid_normalized_view_score = normalized_view_score[i][valid_mask] #(B,N,3)
                valid_normalized_grasp_score= normalized_grasp_score[i][valid_mask]# (B,N, 3, 12, 7)
   
                num_points, _  = valid_sparse_points.size()

                # Compute pairwise distances between dense points and unique points

                distances = torch.cdist(dense_points[i], valid_sparse_points)  # (20000, N)

                # Find indices of k-nearest unique points for each dense point
                _, indices = torch.topk(distances, k=2, largest=False, dim=1)  # (20000, k)

                # Assign scores to dense points based on nearest unique points
                affordance_score = torch.mean(valid_normalized_scores[indices], dim=1)  # (20000,)
                
                approach_directions_reshaped = valid_approach_directions.view(-1, 3)  # (N*3, 3)

                grasp_views_ = template_views.transpose(0, 1).contiguous().unsqueeze(0)
                grasp_views_trans_ = approach_directions_reshaped.transpose(0, 1).contiguous().unsqueeze(0)

                view_inds = knn(grasp_views_, grasp_views_trans_, k=1).squeeze() - 1 #(N*3,1)

                view_inds_reshaped = view_inds.view(-1, 3)#(N,3)
                valid_normalized_view_score = normalized_view_score[i][valid_mask] #(B,N,3)
                row_indices = torch.arange(num_points).unsqueeze(1).expand(-1, 3)
                view_scores = torch.zeros(num_points, NUM_VIEW).to(view_inds_reshaped.device)

                view_scores[row_indices, view_inds_reshaped] = valid_normalized_view_score

                marked_tensor = torch.zeros(num_points, NUM_VIEW, 12, 7).to(view_inds_reshaped.device)
                marked_tensor[row_indices, view_inds_reshaped] = valid_normalized_grasp_score
                final_grasp_scores=marked_tensor
                
                # Find the 3 nearest neighbors for each vector
                # view_inds = knn(grasp_views_, grasp_views_trans_, k=3).squeeze() - 1  # (N*3, 3)

                # view_inds_reshaped = view_inds.view(-1, 3, 3)  # (N, 3, 3)
                # valid_normalized_view_score = normalized_view_score[i][valid_mask]  # (B, N, 3)
                # row_indices = torch.arange(num_points).unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                # view_scores = torch.zeros(num_points, NUM_VIEW).to(view_inds_reshaped.device)

                # # Update the view_scores tensor based on the 3 nearest neighbors for each vector without a loop
                # view_scores[row_indices, view_inds_reshaped] = valid_normalized_view_score.unsqueeze(1).expand(-1, 3, -1)

                # marked_tensor = torch.zeros(num_points, NUM_VIEW, 12, 7).to(view_inds_reshaped.device)
                # marked_tensor[row_indices[:, :, 0], view_inds_reshaped[:, :, 0]] = valid_normalized_grasp_score
                # final_grasp_scores = marked_tensor

                
                #approach_directions_reshaped = valid_approach_directions.view(-1, 3)  # (N*3, 3)
                #normalized_grasp_score=valid_normalized_grasp_score.reshape(num_points * 3, 12, 7) # (N* 3, 12, 7)
                
                #print(approach_directions_reshaped.shape,normalized_grasp_score.shape)
                #################################
                """
                non_zero_indices = torch.nonzero(final_grasp_scores, as_tuple=False)

                #print(non_zero_indices)
                #print(asd)
                #non_zero_indices2 = torch.nonzero(normalized_grasp_score, as_tuple=False)
                #valid_sparse_points_ori = valid_sparse_points.repeat_interleave(3, dim=0)

                #print(non_zero_indices[:,0],non_zero_indices[:,1],non_zero_indices[:,2],non_zero_indices[:,3])
                grasp_center=valid_sparse_points[non_zero_indices[:,0]]
                #grasp_center_expanded = grasp_center.repeat_interleave(3, dim=0)

                #grasp_approach_direction=approach_directions_reshaped[non_zero_indices2[:,0]]
                grasp_approach_direction_ind=non_zero_indices[:,1]
                grasp_angle=((non_zero_indices[:,2]).float())*np.pi/12
                grasp_depth=((non_zero_indices[:,3]).float()+ 1)
                
                #grasp_center=valid_sparse_points_ori[non_zero_indices2[:,0]]
                #grasp_approach_direction=recover_ind[non_zero_indices2[:,0]]
                #grasp_angle=((non_zero_indices2[:,1]).float())*np.pi/12
                #grasp_depth=((non_zero_indices2[:,2]).float()+ 1)
                
                grasp_depth = grasp_depth.view(-1, 1)

                #template_views = generate_grasp_views(self.num_view).to(grasp_center.device)  # (num_view, 3)
                #template_views = template_views.view(self.num_view, 3).contiguous()
                #print(non_zero_indices2.shape,non_zero_indices.shape,grasp_approach_direction_ind.shape)
                grasp_approach_direction=template_views[grasp_approach_direction_ind]
                grasp_rot = batch_viewpoint_params_to_matrix_data(grasp_approach_direction, grasp_angle)

                
                
                
                hand_mesh = o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol3.ply")
 

                point_cloud_pcd = o3d.geometry.PointCloud()
                point_cloud_np = dense_points[i].cpu().numpy().astype(np.float64)
                point_cloud_pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

                grasp_location_spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(location) for location in grasp_center.cpu().numpy()]
                for sphere in grasp_location_spheres:
                    sphere.paint_uniform_color([0, 1, 1])  # Cyan color for grasp location spheres


                gripper_poses = []
                grasp_locations = grasp_center - grasp_approach_direction*grasp_depth*0.01
       
                for rotation, location in zip(grasp_rot.cpu().numpy(), grasp_locations.cpu().numpy()):
                    # Create a transformation matrix from the rotation and location
                    transformation = np.eye(4)
                    transformation[:3, :3] = rotation
                    transformation[:3, 3] = location

                    # Create a copy of the hand mesh and apply the transformation
                    transformed_mesh = copy.deepcopy(hand_mesh).transform(transformation)

                    # Add the transformed mesh to the list
                    gripper_poses.append(transformed_mesh)

                # Visualize the point cloud with the gripper poses
                ##   
                o3d.visualization.draw_geometries([point_cloud_pcd,*grasp_location_spheres,*gripper_poses])
                """
                #################################################################

                distances = torch.cdist(seed_xyz_graspable[i], valid_sparse_points)
                # Find the indices of the nearest neighbors (smallest distances)
                nn_inds = torch.argmin(distances, dim=1)

                # Gather the scores and rotation matrices for the nearest neighbors
                #end_points_view_score[i] = view_scores[nn_inds]
                #end_points_grasp_score[i] = final_grasp_scores[nn_inds]
                angles = torch.zeros(template_views.size(0), dtype=template_views.dtype, device=template_views.device)

                grasp_views_rot = batch_viewpoint_params_to_matrix_data(template_views, angles)  # (V, 3, 3)
                #grasp_views_rot = batch_viewpoint_params_to_matrix(-template_views, angles)  # (V, 3, 3)
                
                grasp_views_rot_trans = torch.matmul(end_points["augument_matrix"][i], grasp_views_rot)  # (V, 3, 3)

                grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(self.M_points, -1, -1,
                                                                              -1)  # (Np, V, 3, 3)
                #end_points_view_rot[b] = valid_view_rot_matrices[nn_inds]
                end_points_view_score[i] = view_scores[nn_inds]
                batch_grasp_point_tensor[i] = valid_sparse_points[nn_inds]
                end_points_grasp_score[i] = final_grasp_scores[nn_inds]
                end_points_view_rot[i] = grasp_views_rot_trans
                end_points_affordance[i]=affordance_score

                # Append to the list for stacking later

                end_points_view_score_list.append(end_points_view_score[i])
                batch_grasp_point_list.append(batch_grasp_point_tensor[i])
                end_points_grasp_score_list.append(end_points_grasp_score[i])
                end_points_view_rot_list.append(end_points_view_rot[i])
                pointwise_label_list.append(end_points_affordance[i])
                #dense_pcd, score_mask_pcd = visualize_with_score_mask(dense_points[i].cpu().numpy(), end_points_affordance[i].cpu().numpy())
                #visualize_vectors_with_scores(valid_sparse_points.cpu().numpy(), view_scores.cpu().numpy(), generate_grasp_views(300).cpu().numpy(), fixed_length=0.1)
            #torch.cuda.empty_cache()
            # Stack the lists to create batch tensors
            batch_grasp_view_graspness = torch.stack(end_points_view_score_list)
            batch_grasp_point = torch.stack(batch_grasp_point_list)
            batch_grasp_score = torch.stack(end_points_grasp_score_list)
            batch_view_rot = torch.stack(end_points_view_rot_list)
            pointwise_label=torch.stack(pointwise_label_list)
            # Store the batch tensors in the end_points dictionary
            end_points['batch_grasp_view_graspness'] = batch_grasp_view_graspness
            end_points['batch_grasp_point'] = batch_grasp_point
            end_points['batch_grasp_score'] = batch_grasp_score
            end_points['batch_grasp_view_rot'] = batch_view_rot
            end_points['pointwise_label'] = pointwise_label

            top_view_inds = end_points['grasp_top_view_inds']  # (B, Ns)
            template_views_rot = end_points['batch_grasp_view_rot']  # (B, Ns, V, 3, 3)
            #print(template_views_rot.shape)
            grasp_scores = copy.deepcopy(end_points['batch_grasp_score'])  # (B, Ns, V, A, D)
            #grasp_widths = end_points['batch_grasp_width']  # (B, Ns, V, A, D, 3)
            # Assuming end_points['batch_grasp_score'] is a PyTorch tensor
            # for b in range(batch_size):
            #      non_zero_indices = torch.nonzero(grasp_scores[b])

            # #     # Check if there are any non-zero elements
            #      has_non_zero = torch.any(grasp_scores[b] != 0)

            #      print("Non-zero indices:", non_zero_indices)
            #      print("Contains non-zero elements:", has_non_zero)

            # Sum over the A and D dimensions to get a score for each (B, Ns, V)
            # summed_scores = grasp_scores.sum(dim=[3, 4])  # (B, Ns, V)

            # # Find the index of the maximum score in the V dimension
            # top_view_inds = summed_scores.argmax(dim=2)  # (B, Ns)
            
            B, Ns, V, A, D = grasp_scores.size()
            top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
            top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)
            top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
            top_view_grasp_scores = torch.gather(grasp_scores, 2, top_view_inds_).squeeze(2)
            #top_view_grasp_widths = torch.gather(grasp_widths, 2, top_view_inds_).squeeze(2)
            #print(top_view_inds_)
            #print(top_view_inds_)
            #u_max = top_view_grasp_scores.max()
            #po_mask = top_view_grasp_scores > 0


            # po_mask_num = torch.sum(po_mask)
            # if po_mask_num > 0:
            #     u_min = top_view_grasp_scores[po_mask].min()
            #     top_view_grasp_scores[po_mask] = torch.log(u_max / top_view_grasp_scores[po_mask]) / (torch.log(u_max / u_min) + 1e-6)

            end_points['batch_grasp_score'] = top_view_grasp_scores  # (B, Ns, A, D)


            # for b in range(batch_size):
            #      non_zero_indices = torch.nonzero(top_view_grasp_scores[b])

            # #     # Check if there are any non-zero elements
            #      has_non_zero = torch.any(grasp_scores[b] != 0)

            #      print("Non-zero indices:", non_zero_indices)
            # #      print("Contains non-zero elements:", has_non_zero)
                 
            grasp_top_views_rot=top_template_views_rot
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        group_features = self.group(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.posead(group_features, end_points)

        return end_points
    


def visualize_with_score_mask(segmented_point_cloud, dense_scores, rgb_colors=None):
    # Normalize the scores for colormap
    normalized_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))
    
    # Get color map from Matplotlib
    cmap = matplotlib.cm.get_cmap('plasma')
    score_colors = cmap(normalized_scores)[:, :3]  # RGB colors based on scores
    
    # Set alpha for blending
    alpha = 0.7
    
    # If rgb_colors is not provided, use uniform gray for all points
    if rgb_colors is None:
        rgb_colors = np.full_like(segmented_point_cloud, fill_value=127)  # Gray
    
    # Blend the score colors with the rgb_colors
    blended_colors = alpha * score_colors + (1 - alpha) * rgb_colors / 255.0
    
    # Create a point cloud object for the dense points
    dense_pcd = o3d.geometry.PointCloud()
    dense_pcd.points = o3d.utility.Vector3dVector(segmented_point_cloud)
    dense_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Uniform gray color
    
    # Create another point cloud object for the blended colors (score mask)
    score_mask_pcd = o3d.geometry.PointCloud()
    score_mask_pcd.points = o3d.utility.Vector3dVector(segmented_point_cloud)
    score_mask_pcd.colors = o3d.utility.Vector3dVector(blended_colors)
    
    # Display the dense points first, then overlay the score mask
    #o3d.visualization.draw_geometries([dense_pcd], window_name="Dense Point Cloud")
    o3d.visualization.draw_geometries([score_mask_pcd], window_name="Score Mask Overlay")

    return dense_pcd, score_mask_pcd


def normalize_grasp_score(grasp_score):
    min_val = torch.min(grasp_score)
    max_val = torch.max(grasp_score)
    grasp_score_normalized = (grasp_score - min_val) / (max_val - min_val)
    grasp_score_normalized = grasp_score_normalized.view(-1, 1)
    return grasp_score_normalized

def combine_scores(grasp_score, closest_scores):
    # Geometric mean combination

        combined_scores = torch.sigmoid(closest_scores) * torch.log1p(grasp_score)

        return combined_scores

def pred_decode(end_points):
    NUM_ANGLE=12
    NUM_DEPTH=7
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for b in range(batch_size):
        grasp_center = end_points['xyz_graspable'][b].float()
        grasp_score = end_points['grasp_score_pred'][b].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        #print(grasp_score.shape, grasp_score_inds)
        
        grasp_score=normalize_grasp_score(grasp_score)
        
        angle_index = grasp_score_inds // NUM_DEPTH
        depth_index = grasp_score_inds % NUM_DEPTH

        # Recover grasp_angle and grasp_depth

        grasp_angle = (angle_index.float())* np.pi / NUM_ANGLE
        grasp_depth = (depth_index.float() + 1) 
        grasp_depth = grasp_depth.view(-1, 1)


        approaching = end_points['grasp_top_view_xyz'][b].float()

        grasp_rot = batch_viewpoint_params_to_matrix_data(approaching, grasp_angle)

        dense_coordinates = end_points['point_clouds'][0, :, 0:3]

        distances = torch.cdist(grasp_center, dense_coordinates)

        # Find the indices of the closest points
        closest_indices = torch.argmin(distances, dim=1)

        # Extract the scores for the closest points
        
        # Convert to NumPy arrays
        grasp_center_np = grasp_center.cpu().numpy().astype(np.float64)
        point_cloud_np = end_points['point_clouds'].cpu().numpy().astype(np.float64)

        # Remove the batch dimension if present
        point_cloud_np = point_cloud_np.squeeze(0)

        # Create Open3D point cloud for the scene
        point_cloud_pcd = o3d.geometry.PointCloud()
        point_cloud_pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])

        # Create lines to represent the vectors

        template_views = generate_grasp_views(NUM_VIEW).to(end_points['point_clouds'].device) 
        selected_views = template_views[end_points['grasp_top_view_inds'][b]]
       
        #print(grasp_depth.shape,grasp_depth)
        grasp_locations = grasp_center - selected_views * grasp_depth*0.01
        #print(grasp_locations)

        # Visualize the point cloud with grasp centers, approach vectors, and grasp locations
        #o3d.visualization.draw_geometries([point_cloud_pcd, *spheres, line_set, *grasp_location_spheres])
            
        # Load the gripper mesh
        hand_mesh = o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol3.ply")

        # Assume grasp_rotations and grasp_locations are already defined
        # grasp_rotations: list of rotation matrices for each grasp pose
        # grasp_locations: list of translation vectors for each grasp pose

        # Create a list to hold the transformed gripper meshes
        gripper_poses = []

        # Transform and add each gripper mesh to the list
        for rotation, location in zip(grasp_rot.cpu().numpy(), grasp_locations.cpu().numpy()):
            # Create a transformation matrix from the rotation and location
            transformation = np.eye(4)
            transformation[:3, :3] = rotation
            transformation[:3, 3] = location

            # Create a copy of the hand mesh and apply the transformation
            transformed_mesh = copy.deepcopy(hand_mesh).transform(transformation)

            # Add the transformed mesh to the list
            gripper_poses.append(transformed_mesh)

        # Visualize the point cloud with the gripper poses
         ##   
       # o3d.visualization.draw_geometries([point_cloud_pcd,*gripper_poses])
            
        #o3d.visualization.draw_geometries([point_cloud_pcd, *spheres, line_set])

        # Visualize the point cloud with grasp centers and lines
        #o3d.visualization.draw_geometries([point_cloud_pcd, *spheres, line_set])


        #print(end_points['graspness_score'][0].shape)
        graspness_score=end_points['graspness_score'][0].cpu().numpy().astype(np.float64)
        closest_scores = end_points['graspness_score'][0][closest_indices]

        #visualize_with_score_mask(point_cloud_np[:, :3],graspness_score.squeeze(1))
        #from grasp_nms import nms_grasp
        #translation_thresh = 0.03
        #rotation_thresh = 30.0 / 180.0 * np.pi
        #nms_grasp(np.array(gripper_poses), translation_thresh, rotation_thresh)
        
        closest_scores=normalize_grasp_score(closest_scores)
        
        
        

        combine_score = combine_scores(grasp_score, closest_scores)

        combine_score=normalize_grasp_score(combine_score)

        grasp_rot=grasp_rot.view(M_POINT, 9)

        obj_ids = -1 * torch.ones_like(grasp_score)

        m=end_points['m']
        centroid=end_points['centroid']
        device = end_points['graspness_score'].device  # the target device

        grasp_center = torch.tensor((grasp_center_np * m) + centroid, dtype=torch.float32).to(device)


        grasp_preds.append(
            torch.cat([combine_score,grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
        

    return grasp_preds


