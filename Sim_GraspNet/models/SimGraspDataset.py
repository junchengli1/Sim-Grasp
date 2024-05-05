#import open3d as o3d
import glob
import pickle
import numpy as np
import sys
import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
import open3d as o3d
import provider
import torch

import os
import numpy as np

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
#import MinkowskiEngine as ME
import h5py
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture
import matplotlib.cm
from scipy.spatial import cKDTree
from collections import defaultdict
import copy
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

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    #centroid=0
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    #pc=pc
   # m=0
    return pc, centroid, m

def make_one_hot(labels, threshold=0.5):
    labels = np.where(labels > threshold, 1, 0)
    return labels
def one_hot_max(input_array, threshold=0.2):
    # First, set all values below the threshold to 0
    input_array[input_array < threshold] = 0

    # Then, set the max value in each row to 1, and all other values to 0
    row_max_indices = np.argmax(input_array, axis=1)
    one_hot_array = np.zeros_like(input_array)
    one_hot_array[np.arange(input_array.shape[0]), row_max_indices] = 1

    # Finally, zero out rows that were all zero in the input array
    one_hot_array[np.all(input_array == 0, axis=1)] = 0

    return one_hot_array
def rotate_point_cloud_with_normal(xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            xyz_normal: N,6, first three channels are XYZ, last 3 all normal
        Output:
            N,6, rotated XYZ, normal point cloud
    '''
    rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
    #rot_angle = 0  # -30 ~ +30 degree

    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, c, -s],
                                [0, s, c]])
    
    shape_pc = xyz_normal[:, 0:3]
    shape_normal = xyz_normal[:, 3:6]
    
    xyz_normal[:, 0:3] = np.dot(shape_pc, rotation_matrix)
    xyz_normal[:, 3:6] = np.dot(shape_normal, rotation_matrix)

    return xyz_normal,rotation_matrix
def rotate_point_cloud_by_angle_with_push_vectors(data):
    """ 
    Rotate the point cloud along up direction with random angle, including normal and push vectors.
    Input:
      Nx24 array, original point cloud with normal and push vectors
    Return:
      Nx24 array, rotated point cloud with normal and push vectors
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    rotation_angle =  np.random.uniform() * np.pi / 4
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    shape_pc = data[:,0:3]
    shape_normal = data[:,3:6]
    shape_push_vectors = data[:,6:24].reshape(-1, 6, 3)  # reshape to 3D vectors
    #print("Shape of shape_push_vectors before rotation:", shape_push_vectors.shape)
    rotated_push_vectors = np.zeros_like(shape_push_vectors)
    for i in range(6):  # assuming you have 6 3D vectors
        rotated_push_vectors[:, i, :] = np.dot(shape_push_vectors[:, i, :], rotation_matrix)
    #print("Shape of rotated_push_vectors after rotation:", rotated_push_vectors.shape)
    rotated_data[:,6:24] = rotated_push_vectors.reshape(-1, 18) 

    rotated_data[:,0:3] = np.dot(shape_pc, rotation_matrix)
    rotated_data[:,3:6] = np.dot(shape_normal, rotation_matrix)
    #rotated_data[:,6:24] = np.dot(shape_push_vectors, rotation_matrix).reshape(-1, 18)  # reshape back to original
    #rotated_data[:,6:24] = np.dot(shape_push_vectors, rotation_matrix).reshape(-1, 18)  # reshape back to original
    return rotated_data
def score_to_color(score, min_score=0.5, max_score=1.0):
    # Linearly interpolate between blue (low) and red (high)
    color_low = np.array([0, 0, 1])  # Blue
    color_high = np.array([1, 0, 0])  # Red
    
    # Normalize score to [0, 1]
    normalized_score = (score - min_score) / (max_score - min_score)
    
    # Interpolate color based on normalized score
    color = (1 - normalized_score) * color_low + normalized_score * color_high
    return color
def random_scale_point_cloud(data, scale_low=0.9, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    N, C = data.shape
    scales = np.random.uniform(scale_low, scale_high)
    #for batch_index in range(n):
    data[:,:] *= scales
    return data,scales
def visualize_point_clouds(scored_positions, scores, points_with_features):
    # Create a point cloud object for scored_positions
    scored_pcd = o3d.geometry.PointCloud()
    scored_pcd.points = o3d.utility.Vector3dVector(scored_positions)
    
    # Map scores to colors using a colormap (e.g., hot, which goes from blue to red)
    max_score = scores.max()
    min_score = scores.min()
    scores_normalized = (scores - min_score) / (max_score - min_score)
    colors = plt.get_cmap('hot')(scores_normalized)[:, :3]  # Get the RGB values from the colormap
    scored_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a point cloud object for points_with_features
    features_pcd = o3d.geometry.PointCloud()
    features_pcd.points = o3d.utility.Vector3dVector(points_with_features[:, :3])
    
    # Assuming the last three columns are RGB values for visualization
    # Normalize feature values to [0, 1] for color mapping if necessary
    features_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Visualize both point clouds in the same window
    o3d.visualization.draw_geometries([scored_pcd, features_pcd], 
                                      window_name="Scored Positions and Point Cloud with Features",
                                      point_show_normal=False)
def visualize_dense_scores_heatmap(points, dense_scores):
    """
    Visualize a heatmap of dense scores on a point cloud.

    :param points: Nx3 numpy array of point cloud positions.
    :param dense_scores: N-length numpy array of scores for each point.
    """
    # Normalize the dense scores to [0, 1] for colormap mapping
    scores_normalized = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))
    
    # Apply a colormap (e.g., 'hot') to the normalized scores to get RGB colors
    cmap = plt.get_cmap('hot')
    colors = cmap(scores_normalized.flatten())[:, :3]  # Get the RGB values, ignore alpha

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize the point cloud with the heatmap
    o3d.visualization.draw_geometries([pcd], window_name="Dense Scores Heatmap")
def visualize_binary_grasp_quality_heatmap(points, binary_scores):
    # Create a custom color map for binary scores: 0s to red, 1s to green
    colors = np.zeros((len(binary_scores), 3))
    colors[binary_scores == 0, :] = [1, 0, 0]  # Red for non-graspable points
    colors[binary_scores == 1, :] = [0, 1, 0]  # Green for graspable points

    # Create an Open3D point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with the heatmap
    o3d.visualization.draw_geometries([pcd], window_name="Binary Grasp Quality Heatmap")

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
def score_to_color(score, min_score, max_score):
    # Normalize score to [0, 1]
    normalized_score = (score - min_score) / (max_score - min_score)
    # Interpolate color: red (low) to green (high)
    return [1-normalized_score, normalized_score, 0]  # RGB color
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
def print_grasp_structure_info(grasp_data_structure):
    # Top Level - Grasp Points
    print(f"Total number of unique grasp points (t_ori): {len(grasp_data_structure)}")

    for point_key, approach_vectors in grasp_data_structure.items():
        print(f"\nGrasp Point {point_key}:")
        print(f"  Number of unique approach vectors: {len(approach_vectors)}")

        for approach_key, rotation_matrices in approach_vectors.items():
            print(f"    Approach Vector {approach_key}:")
            print(f"      Number of unique suction rotation matrices: {len(rotation_matrices)}")

            for rotation_key, depths in rotation_matrices.items():
                print(f"        Rotation Matrix {rotation_key}:")
                print(f"          Number of stand-off depths: {len(depths)}")


def preprocess_grasp_data(candidate_simulation):
    # Calculate the number of unique t_ori points
    total_grasp_poses = sum(len(grasp_data["grasp_samples"]) for grasp_data in candidate_simulation.values())
    num_unique_points = total_grasp_poses // (3 * 12 * 7)

    # Assuming maximum values for approach directions, rotation angles, and depths
    max_approach_directions = 3
    max_rotation_angles = 12
    max_depths = 7

    # Initialize the arrays
    approach_directions = np.zeros((num_unique_points, max_approach_directions, 3))
    #rotation_matrices = np.zeros((num_unique_points, max_approach_directions, max_rotation_angles, 3, 3))
    #stand_off_depths = np.zeros((num_unique_points, max_approach_directions, max_rotation_angles, max_depths))
    # Organize the data
    affordance_scores = np.zeros((num_unique_points, max_approach_directions, max_rotation_angles, max_depths))
    view_score = np.zeros((num_unique_points, max_approach_directions, max_rotation_angles, max_depths))

    # Extract unique t_ori and their corresponding samples
    t_ori_dict = defaultdict(list)
    for object_index, grasp_data in candidate_simulation.items():
        for sample in grasp_data["grasp_samples"]:
            t_ori = tuple(sample["t_ori"][:3])
            t_ori_dict[t_ori].append(sample)
            #print(sample)

    # Populate the arrays
    for i, (t_ori, samples) in enumerate(t_ori_dict.items()):
            for j, sample in enumerate(samples):
                # Calculate indices based on the nested structure
                approach_dir_index = j // (max_rotation_angles * max_depths) % max_approach_directions
                rotation_index = (j // max_depths) % max_rotation_angles
                stand_off_index = j % max_depths
                approach_directions[i, approach_dir_index] = sample['approach_direction']
                #rotation_matrices[i, approach_dir_index, rotation_index] = sample['grasp_rotation_matrix']
                #stand_off_depths[i, approach_dir_index, rotation_index, stand_off_index] = sample['stand_off']
  
                if sample.get("simulation_quality", 0) == 1:
                    affordance_scores[i, approach_dir_index, rotation_index, stand_off_index] = 1
                    view_score[i, approach_dir_index, rotation_index, stand_off_index] = 10
                elif sample.get("collision_quality", 0) != 0:
                    view_score[i, approach_dir_index, rotation_index, stand_off_index] = sample.get("collision_quality", 0)
                    #view_score[i, approach_dir_index, rotation_index, stand_off_index] = 0

        # Sum the scores for each point across the subdimensions (3, 12, 7)
    summed_scores = np.sum(affordance_scores, axis=(1, 2, 3))

    # Normalize the summed scores from 0 to 1 based on their minimum and maximum values
    min_score = np.min(summed_scores)
    max_score = np.max(summed_scores)
    range_score = max_score - min_score
    normalized_scores = (summed_scores - min_score) / range_score
    unique_t_ori_points = np.array(list(t_ori_dict.keys()))

    summed_view_score = np.sum(view_score, axis=(2, 3)) 
    #print(summed_view_score)
    min_values_per_channel = np.min(summed_view_score, axis=1, keepdims=True)
    max_values_per_channel = np.max(summed_view_score, axis=1, keepdims=True)
    range_values_per_channel = max_values_per_channel - min_values_per_channel

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    normalized_summed_view_score = (summed_view_score - min_values_per_channel) / (range_values_per_channel + epsilon)

   # print(normalized_summed_view_score)

    return unique_t_ori_points,normalized_scores,approach_directions,normalized_summed_view_score,affordance_scores
def rotation_matrices_from_vectors(vecs1, vecs2):
    """Find the rotation matrices that align vecs1 to vecs2."""
    a = vecs1 / np.linalg.norm(vecs1, axis=-1, keepdims=True)
    b = vecs2 / np.linalg.norm(vecs2, axis=-1, keepdims=True)
    v = np.cross(a, b)
    c = np.einsum('ij,ij->i', a, b.reshape(1, -1))  # Reshape vecs2 for broadcasting
    s = np.linalg.norm(v, axis=-1)
    kmat = np.cross(np.eye(3)[np.newaxis, :, :], v[:, np.newaxis, :])  # Ensure kmat has shape (N, 3, 3)
    rotation_matrices = np.eye(3) + kmat + np.einsum('ijk,ikl->ijl', kmat, kmat) * ((1 - c) / (s ** 2 + 1e-8))[:, np.newaxis, np.newaxis]
    return rotation_matrices

class simgraspdata2(Dataset):
    def __init__(self, data_root,label_root):
        self.data_root = data_root
        self.label_root = label_root
        #print(data_root)
        #stage_root = os.listdir(data_root)
        
        self.room_points={}
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        sample_rate=0.1
        block_size=1000
        self.block_size=block_size
        self.num_point=40000
        num_point=self.num_point
        exclusion_list = [109,176,378,349,367,474]

        points=np.load(self.data_root+f"/{0}.npz",allow_pickle=True)['arr_0']
        for scene in range(0,500):
            if scene in exclusion_list:
                continue   
            num_point_all.append(points.size)
 
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        filtered_stage_root = [i for i in range(500) if i not in exclusion_list]
        room_idxs = [index for index in filtered_stage_root for _ in range(int(round(sample_prob[filtered_stage_root.index(index)] * num_iter)))]
        self.room_idxs = np.array(room_idxs)

        print("Totally {} samples in {} set.".format(len(self.room_idxs), "train"))
     
        #print(self.room_idxs)

        # Compute the rotation matrices for aligning the view vectors with the z-axis


    def __len__(self):
        return len(self.room_idxs)

    def __getitem__(self, index):
        room_idx = self.room_idxs[index]
        points=np.load(self.data_root+f"/{room_idx}.npz",allow_pickle=True)['arr_0']

        N_points = points.shape[0]    
        center = points[np.random.choice(N_points)][:3]
        block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]  
        
        # Load preprocessed data
        preprocessed_data_path = self.label_root + f"/stage_{room_idx}/stage_{room_idx}_preprocessed_data.pkl"
        with open(preprocessed_data_path, 'rb') as f:
            preprocessed_data = pickle.load(f)

        # Unpack preprocessed data
        unique_t_ori_points = copy.deepcopy(preprocessed_data['unique_t_ori_points'])
        approach_directions = copy.deepcopy(preprocessed_data['approach_directions'])# (N, 3, 3)

        # Assuming N is the number of unique points
        N = len(unique_t_ori_points)

        point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                #if point_idxs.size > 10000:
                # break
        
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)

        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # Select points based on some indices (assuming selected_point_idxs is defined)
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        
        # Scale the selected points
        scaled_points, scale_factor = random_scale_point_cloud(selected_points[:, 0:3])
        
        # Integrate scaled points back with their features
        points_w_feature = np.zeros((self.num_point, 6))
        points_w_feature[:, 0:3] = scaled_points
        points_w_feature[:, 3:6] = selected_points[:, 3:6]  # Assuming features are in columns 3 to 5
        
        # Rotate the points with features
        rotated_points_w_feature, rotation_matrix = rotate_point_cloud_with_normal(points_w_feature)
        
        # Normalize the rotated points
        normalized_points, centroid, max_distance = pc_normalize(rotated_points_w_feature[:, 0:3])
 
        sparse_points_xyz=unique_t_ori_points

        sparse_points_xyz*= scale_factor

        sparse_points_xyz_rotated = np.dot(unique_t_ori_points, rotation_matrix)
        
        # Reshape approach_directions, apply the rotation matrix, and reshape back
        approach_directions_reshaped = approach_directions.reshape(-1, 3)
        rotated_approach_directions_reshaped = np.dot(approach_directions_reshaped, rotation_matrix)
        rotated_approach_directions = rotated_approach_directions_reshaped.reshape(N, 3, 3)
        #rotated_approach_directions = np.dot(unique_t_ori_points, rotation_matrix)

               
        sparse_points_xyz_rotated-= centroid
        sparse_points_xyz_rotated/= max_distance
        sparse_points_xyz_normalized= sparse_points_xyz_rotated
        

        # Integrate normalized points back with their features for the original point cloud
        final_points_w_feature = np.zeros_like(points_w_feature)
        final_points_w_feature[:, 0:3] = normalized_points
        final_points_w_feature[:, 3:6] = rotated_points_w_feature[:, 3:6]  # Keep original features


   
        # grasp_location_spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(location) for location in preprocessed_data['unique_t_ori_points']]
        # for sphere in grasp_location_spheres:
        #     sphere.paint_uniform_color([0, 1, 1])

        # point_cloud_pcd = o3d.geometry.PointCloud()
        # point_cloud_np =points[:,:3]
        # point_cloud_pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        # o3d.visualization.draw_geometries([point_cloud_pcd,*grasp_location_spheres])

        # visualize_with_score_mask(final_points_w_feature[:,0:3], normalized_dense_scores, rgb_colors=None)
      
        binary_view_scores = (preprocessed_data['normalized_view_score']>0.9)
       

        ret_dict = {'point_clouds': final_points_w_feature.astype(np.float64),
                    "coors":final_points_w_feature[:,0:3].astype(np.float64),
                    "feats":final_points_w_feature[:,3:6].astype(np.float64),
                    'sparse_points': sparse_points_xyz_normalized.astype(np.float64),
                    'normalized_scores': preprocessed_data['normalized_scores'].astype(np.float64),
                    'rotated_approach_directions': rotated_approach_directions.astype(np.float64),
                    'normalized_view_score': binary_view_scores.astype(np.float64),
                    'normalized_grasp_score': preprocessed_data['normalized_grasp_score'].astype(np.float64), 
                    "augument_matrix":rotation_matrix,
                    }
        
        return ret_dict




class simgraspdata(Dataset):
    def __init__(self, data_root,label_root):
        self.data_root = data_root
        self.label_root = label_root
        #print(data_root)
        #stage_root = os.listdir(data_root)
        
        self.room_points={}
        self.room_coord_min, self.room_coord_max = [], []
        
        block_size=1000
        self.block_size=block_size
        self.num_point=40000
        num_point=self.num_point
        exclusion_list = [109,176,378,349,367,474]

       # Initialize a dictionary to store the count of subframes for each stage
        stage_frame_count = {}
        self.data_list = []

        for file_name in os.listdir(self.data_root):
            if file_name.endswith('.npz'):
                parts = file_name.split('_')
                if len(parts) == 2:
                    # This is a frame point cloud file
                    stage, frame_with_ext = parts
                    frame = frame_with_ext.split('.')[0]
                    stage = int(stage)
                    if stage not in exclusion_list:
                        self.data_list.append([stage, int(frame)])  # Store the stage and frame
                        if stage not in stage_frame_count:
                            stage_frame_count[stage] = 1
                        else:
                            stage_frame_count[stage] += 1
                elif len(parts) == 1:
                    # This is a multiview point cloud file
                    stage_with_ext = parts[0]
                    stage = int(stage_with_ext.split('.')[0])
                    if stage not in exclusion_list:
                        self.data_list.append([stage, None])  # Store the stage and indicate no frame
                        if stage not in stage_frame_count:
                            stage_frame_count[stage] = 1
                        else:
                            stage_frame_count[stage] += 1

        # Calculate the total length as the sum of all stages and their respective subframes
        self.total_length = sum(stage_frame_count.values())
        print("Total length:", self.total_length)
        #print(self.room_idxs)

        # Compute the rotation matrices for aligning the view vectors with the z-axis


    def __len__(self):
        return (self.total_length)

    def __getitem__(self, index):

        room_idx, frame = self.data_list[index]

        if frame is not None:
            file_path = f"{self.data_root}/{room_idx}_{frame}.npz"
        else:
            file_path = f"{self.data_root}/{room_idx}.npz"

        points = np.load(file_path, allow_pickle=True)['arr_0']

        N_points = points.shape[0]    
        center = points[np.random.choice(N_points)][:3]
        block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]  
        
        # Load preprocessed data
        preprocessed_data_path = self.label_root + f"/stage_{room_idx}/stage_{room_idx}_preprocessed_data.pkl"
        with open(preprocessed_data_path, 'rb') as f:
            preprocessed_data = pickle.load(f)

        # Unpack preprocessed data
        unique_t_ori_points = copy.deepcopy(preprocessed_data['unique_t_ori_points'])
        approach_directions = copy.deepcopy(preprocessed_data['approach_directions'])# (N, 3, 3)

        # Assuming N is the number of unique points
        N = len(unique_t_ori_points)

        point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                #if point_idxs.size > 10000:
                # break
        
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)

        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # Select points based on some indices (assuming selected_point_idxs is defined)
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        
        # Scale the selected points
        scaled_points, scale_factor = random_scale_point_cloud(selected_points[:, 0:3])
        
        # Integrate scaled points back with their features
        points_w_feature = np.zeros((self.num_point, 6))
        points_w_feature[:, 0:3] = scaled_points
        points_w_feature[:, 3:6] = selected_points[:, 3:6]  # Assuming features are in columns 3 to 5
        
        # Rotate the points with features
        rotated_points_w_feature, rotation_matrix = rotate_point_cloud_with_normal(points_w_feature)
        
        # Normalize the rotated points
        normalized_points, centroid, max_distance = pc_normalize(rotated_points_w_feature[:, 0:3])
 
        sparse_points_xyz=unique_t_ori_points

        sparse_points_xyz*= scale_factor

        sparse_points_xyz_rotated = np.dot(unique_t_ori_points, rotation_matrix)
        
        # Reshape approach_directions, apply the rotation matrix, and reshape back
        approach_directions_reshaped = approach_directions.reshape(-1, 3)
        rotated_approach_directions_reshaped = np.dot(approach_directions_reshaped, rotation_matrix)
        rotated_approach_directions = rotated_approach_directions_reshaped.reshape(N, 3, 3)
        #rotated_approach_directions = np.dot(unique_t_ori_points, rotation_matrix)

               
        sparse_points_xyz_rotated-= centroid
        sparse_points_xyz_rotated/= max_distance
        sparse_points_xyz_normalized= sparse_points_xyz_rotated
        

        # Integrate normalized points back with their features for the original point cloud
        final_points_w_feature = np.zeros_like(points_w_feature)
        final_points_w_feature[:, 0:3] = normalized_points
        final_points_w_feature[:, 3:6] = rotated_points_w_feature[:, 3:6]  # Keep original features


   
        # grasp_location_spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=1).translate(location) for location in preprocessed_data['unique_t_ori_points']]
        # for sphere in grasp_location_spheres:
        #     sphere.paint_uniform_color([0, 1, 1])

        # point_cloud_pcd = o3d.geometry.PointCloud()
        # point_cloud_np =points[:,:3]
        # point_cloud_pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        # o3d.visualization.draw_geometries([point_cloud_pcd,*grasp_location_spheres])

        # visualize_with_score_mask(final_points_w_feature[:,0:3], normalized_dense_scores, rgb_colors=None)
      
        binary_view_scores = (preprocessed_data['normalized_view_score']>0.9)
       

        ret_dict = {'point_clouds': final_points_w_feature.astype(np.float64),
                    "coors":final_points_w_feature[:,0:3].astype(np.float64),
                    "feats":final_points_w_feature[:,3:6].astype(np.float64),
                    'sparse_points': sparse_points_xyz_normalized.astype(np.float64),
                    'normalized_scores': preprocessed_data['normalized_scores'].astype(np.float64),
                    'rotated_approach_directions': rotated_approach_directions.astype(np.float64),
                    'normalized_view_score': binary_view_scores.astype(np.float64),
                    'normalized_grasp_score': preprocessed_data['normalized_grasp_score'].astype(np.float64), 
                    "augument_matrix":rotation_matrix,
                    }
        
        return ret_dict













