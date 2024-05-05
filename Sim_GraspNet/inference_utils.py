import open3d as o3d
import os
import sys
import numpy as np
import copy


class CollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
    """
    def __init__(self, scene_points,voxel_size=0.5):
        #gg_array=gg_array
        self.voxel_size = voxel_size
        #self.voxel_size =0.8
        self.scene_cloud = o3d.geometry.PointCloud()
        self.scene_cloud.points = o3d.utility.Vector3dVector(scene_points)

        scene_cloud_down = self.scene_cloud.voxel_down_sample(self.voxel_size)
        #self.scene_points = np.array(scene_cloud.points)
        #o3d.visualization.draw_geometries([scene_cloud])
        #self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(scene_cloud, 0.5)
        #self.scene_points = np.array(self.voxel_grid.points)
        #scene_cloud = scene_cloud.voxel_down_sample(self.voxel_size)
        self.scene_points = np.array(scene_cloud_down.points)
        #o3d.visualization.draw_geometries([scene_cloud_down])
        #self.point_cloud=scene_cloud

    def detect(self, gg_array, approach_dist=0, collision_thresh=0, return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """
      
        """
          # Update with your gripper's width
        #T = grasp_group.translations
        #approach_dist=0
        #rotation_ori = grasp_group.rotation_matrices
        rotation=gg_array[:, 2:11].reshape(-1, 3, 3)
        depths_reshaped = gg_array[:, 1].reshape(-1, 1)
        #print(gg_array.shape)
        grasp_center = gg_array[:, 11:14]
        #center=grasp_center
        center = grasp_center - rotation[:,:,2] * depths_reshaped

        targets = self.scene_points[np.newaxis, :, :] - center[:, np.newaxis, :]
        #print(targets.shape,rotation.shape)

        # Loop through each grasp and apply the rotation
        transformed_targets=np.empty_like(targets)
        for i in range(targets.shape[0]):
            transformed_targets[i] = np.matmul(targets[i], rotation[i])
        targets=transformed_targets
        # Update targets with the transformed values
        #targets = transformed_targets[:, np.newaxis, 2]+ depths_reshaped[:, np.newaxis, :]

        # collision detection
        # left finger mask
        mask_left = ((targets[:, :, 0] > (-1.4 - 7.83/2)) & (targets[:, :, 0] < ( - 7.83/2)) &
                    (targets[:, :, 1] > -3/2) & (targets[:, :, 1] < 3/2) &
                    (targets[:, :, 2] > 6/2 - 6/2) & (targets[:, :, 2] < 6/2 + 6/2))

        # right finger mask
        mask_right = ((targets[:, :, 0] > (7.83/2) ) & (targets[:, :, 0] < ( 1.4+ 7.83/2) ) &
                    (targets[:, :, 1] > -3/2) & (targets[:, :, 1] < 3/2) &
                    (targets[:, :, 2] > 6/2 - 6/2) & (targets[:, :, 2] < 6/2 + 6/2))

        # box middle mask
        mask_middle = ((targets[:, :, 0] > -13/2) & (targets[:, :, 0] < 13/2) &
                    (targets[:, :, 1] > -7.4/2) & (targets[:, :, 1] < 7.4/2) &
                    (targets[:, :, 2] > 12 - 12/2) & (targets[:, :, 2] < 12 + 12/2))
        
           # shifting mask
        # mask_shifting = ((targets[:, :, 0] > -13/2) & (targets[:, :, 0] < 13/2) &
        #              (targets[:, :, 1] > -7.4/2) & (targets[:, :, 1] < 7.4/2) &
        #              (targets[:, :, 2] > 12 - 12/2 - approach_dist) & (targets[:, :, 2] < 12 - 12/2))
        
        # get collision mask of each point
        #global_mask = (mask_left | mask_right | mask_middle)#| mask_shifting)
        global_mask = (mask_left | mask_right | mask_middle)

        # calculate equivalent volume of each part

        #left_right_volume = np.array(1.4 * 2.5 * 6 / (self.voxel_size**3)).reshape(-1)
        #middle_volume = np.array(13 * 7.4 * 12 / (self.voxel_size**3)).reshape(-1)

        left_right_volume = np.array(1.4 * 2.5 * 6 / (self.voxel_size**3)).reshape(-1)
        middle_volume = np.array(13 * 7.4 * 12 / (self.voxel_size**3)).reshape(-1)
        #shifting_volume = np.array(13 * 7.4 * approach_dist / (self.voxel_size**3)).reshape(-1)

        #shifting_volume = np.array(13 * 7.4 * approach_dist / (self.voxel_size**3)).reshape(-1)
        volume = left_right_volume*2 + middle_volume#+shifting_volume
        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)
        #global_iou = global_mask.sum(axis=1) / (global_mask.shape[1] + 1e-6)
        #print(global_iou)
        print(global_iou)
        collision_mask = (global_iou > collision_thresh)

        # get collision mask
        #visualize=True
        #hand_mesh = o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol2.ply")
        
        # if visualize:
        #     # Create a point cloud object
        #     #pcd = o3d.geometry.PointCloud()
        #     #pcd.points = o3d.utility.Vector3dVector(self.scene_points)

        #     # Create a visualization window
        #     #vis = o3d.visualization.Visualizer()
        #     #vis.create_window()

        #     # Add the point cloud to the visualization
        #     #vis.add_geometry(pcd)

        #     # Iterate over each grasp in the grasp group
        #     #hand_mesh = o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol2.ply")
        #     gripper_poses = []
        #     for grasp_num, grasp in enumerate(grasp_group):
        #         #print(grasp_num)
        #         #T = center[grasp_num,:]
        #         #R = rotation[grasp_num,:]
        #         # Define the transformation matrix for the base of the gripper
        #         origin_T = np.eye(4)
        #         origin_T[:3, :3] = rotation[grasp_num,:]
        #         origin_T[:3, 3] = center[grasp_num,:]

        #         # Define local bounding boxes for each part of the gripper
        #           # Set the color based on the collision status
        #         if collision_mask[grasp_num]:
        #             color = [1, 0, 0]  # Red for collision
        #             #transformed_mesh = copy.deepcopy(hand_mesh).transform(origin_T)
        #             #transformed_mesh.paint_uniform_color(color)

        #             #gripper_poses.append(transformed_mesh)
                    
        #         else:
        #             #print(global_iou[grasp_num])
        #             color = [0, 1, 0]  # Green for no collision
        #             transformed_mesh = copy.deepcopy(hand_mesh).transform(origin_T)
        #             transformed_mesh.paint_uniform_color(color)

        #             gripper_poses.append(transformed_mesh)
        #         #transformed_mesh = copy.deepcopy(hand_mesh).transform(origin_T)
        #         #transformed_mesh.paint_uniform_color(color)

        #         #gripper_poses.append(transformed_mesh)
        #         # if ~collision_mask[grasp_num]:
        #         # #hand_mesh = o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol2.ply")
        #         #     transformed_mesh = copy.deepcopy(hand_mesh).transform(origin_T)

        #         #     # Add the global bounding box to the visualization
        #         #     gripper_poses.append(transformed_mesh)
        #            #vis.add_geometry(transformed_mesh)
        #     # Run the visualization
        #     #vis.run()
        #     #vis.destroy_window()
        #     o3d.visualization.draw_geometries([self.scene_cloud,*gripper_poses])

       
        #print(collision_thresh,collision_mask)
        if not (return_empty_grasp or return_ious):
            return collision_mask

        
def plot_gripper(hand_mesh,center, R, depth, score=1, color=None):
    #hand_mesh = o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol2.ply")
    grasp_locations = center - R[:, 2] * depth
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = grasp_locations
    color = [1 - score, score, 0]  # red for high score
    transformed_mesh = copy.deepcopy(hand_mesh).transform(transformation)
    transformed_mesh.paint_uniform_color(color)
    return transformed_mesh

def sort_grasps_by_score(grasps):
    return grasps[np.argsort(-grasps[:, 0])]

def translation_distance(center1, center2):
    return np.linalg.norm(center1 - center2)

def rotation_distance(rot1, rot2):
    rot_diff = np.matmul(rot1, rot2.T)
    trace = np.trace(rot_diff)
    return np.arccos((trace - 1) / 2)

def grasp_nms_with_object_id(grasps, thd, tha):
    unique_object_ids = np.unique(grasps[:, -1])
    keep = []

    for obj_id in unique_object_ids:
        # Get grasps belonging to the current object
        obj_grasps = grasps[grasps[:, -1] == obj_id]
        
        # Sort grasps based on confidence scores in descending order
        sorted_indices = np.argsort(-obj_grasps[:, 0])
        obj_keep = []
        
        while len(sorted_indices) > 0:
            # Select the grasp with the highest confidence score
            current_index = sorted_indices[0]
            obj_keep.append(current_index)
            
            #if len(obj_keep) == k:
                #break
            
            # Calculate translation and rotation distances between the current grasp and the remaining grasps
            center1 = obj_grasps[current_index, 11:14]
            rot1 = obj_grasps[current_index, 2:11].reshape(3, 3)
            
            remaining_indices = sorted_indices[1:]
            centers2 = obj_grasps[remaining_indices, 11:14]
            rots2 = obj_grasps[remaining_indices, 2:11].reshape(-1, 3, 3)
            
            trans_dists = np.linalg.norm(centers2 - center1, axis=1)
            rot_dists = np.arccos((np.trace(np.matmul(rots2, rot1.T), axis1=1, axis2=2) - 1) / 2)
            
            # Remove grasps that are too close to the current grasp
            keep_indices = np.where((trans_dists >= thd) | (rot_dists >= tha))[0]
            sorted_indices = remaining_indices[keep_indices]
        
        # Map the kept indices back to the original grasp array
        obj_keep = np.where(grasps[:, -1] == obj_id)[0][obj_keep]
        keep.extend(obj_keep)
    
    return keep

 #torch.cat([grasp_score,grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))