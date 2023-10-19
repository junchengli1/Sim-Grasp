# Omniverse Libraries
from omni.isaac.kit import SimulationApp
from pathlib import Path
import argparse
base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='Sim-Suction create_point_cloud_and_seal_evaluation')
parser.add_argument('--headless', type=bool, default=True, help='headless')#make it True when gerneate dataset 
parser.add_argument('--debug_draw', type=bool, default=True, help='debug draw')#make it False when gerneate dataset 
parser.add_argument('--pcl_path', type=str, default=(base_dir.parent / "pointcloud_train").as_posix(), help='point cloud path')
parser.add_argument('--data_path', type=str, default=(base_dir.parent / "synthetic_data").as_posix(), help='data path')
parser.add_argument('--instanceable_flag', type=bool, default=False, help='use textureless instanceable usd to increase simulation speed')
parser.add_argument('--seg_dic_path', type=str, default=(base_dir.parent / "seg_dic.pkl").as_posix(), help='seg_dic path')
parser.add_argument('--save_pcd_flag', type=bool, default=False, help='Save each objects as pcd file')
parser.add_argument('--save_pkl_flag', type=bool, default=True, help='Save seal evaluation as pkl file')
parser.add_argument('--save_pcl_flag', type=bool, default=True, help='Save merged point cloud as npz file')
parser.add_argument('--suction_radius', type=float, default=1.5, help='suction radius 1.5cm')
parser.add_argument('--deformation_thredshold', type=float, default=0.15, help='15 percent deformation')
parser.add_argument('--start_stage', type=int,default=0, help='start stage number')
parser.add_argument('--end_stage', type=int,default=500, help='end stage number')


args = parser.parse_args()
simulation_app = SimulationApp({"headless": args.headless})

from omni.isaac.core import World
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.window.viewport")  # enable legacy viewport interface
import random  # Ensure you have this import at the top of your file

from omni.physx.scripts import utils
import omni
import omni.physx
import omni.replicator.core as rep
import carb
   
# Pixar USD Libraries
from pxr import Usd, UsdGeom

# Standard Libraries
import os
import pickle
import math as m
import asyncio

# Third Party Libraries
import numpy as np
import open3d as o3d
import torch
import torch.utils.data

from dgl.geometry import farthest_point_sampler

from simulation_utils import *  # This imports all functions from the utility file
import trimesh
import trimesh.transformations as tra
import logging
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.rotations import euler_angles_to_quat,matrix_to_euler_angles

def sample_spherical_cap(original_dir,cone_dirs, cone_aperture, num_samples_per_dir=1):
    """Uniformly distributed points on a spherical cap (sphere radius = 1).
    Args:
        cone_dirs (np.array): Nx3 array that represents cone directions.
        cone_aperture (float): Aperture of cones / size of spherical cap.
        num_samples_per_dir (int, optional): Number of samples to draw per direction. Defaults to 1.
    Raises:
        NotImplementedError: [description]
    Returns:
        np.array: Nx3 array of sampled points.
    """
    # sample around north pole
    if num_samples_per_dir > 1:
        raise NotImplementedError("num_samples_per_dir > 1 is not implemented")
    num_samples = len(cone_dirs) * num_samples_per_dir
    x = np.random.rand(num_samples) * (1.0 - np.cos(cone_aperture)) + np.cos(
        cone_aperture
    )
    phi = np.random.rand(num_samples) * 2.0 * np.pi
    y = np.sqrt(1.0 - np.power(x, 2)) * np.cos(phi)
    z = np.sqrt(1.0 - np.power(x, 2)) * np.sin(phi)

    points = np.vstack([x, y, z]).T
    points = points[..., np.newaxis]
    transforms = np.array(
        [
            trimesh.geometry.align_vectors([1,0,0], cone_dir)[:3, :3]
            for cone_dir in cone_dirs
        ]
    )
    #print(points)
    result = np.matmul(transforms, points)
    return np.squeeze(result, axis=2)
def rotation_matrix_from_axis_and_angle(axis, angle):
    """
    Compute a rotation matrix from an axis and an angle.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def rotate_matrix_about_axis(matrix, axis, theta):
    """
    Rotate a matrix about a specified axis by theta degrees
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    rotation_matrix = np.array([
        [cos_theta + axis[0]**2 * (1 - cos_theta), axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta, axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],
        [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta, cos_theta + axis[1]**2 * (1 - cos_theta), axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],
        [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta, axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta, cos_theta + axis[2]**2 * (1 - cos_theta)]
    ])

    return np.dot(rotation_matrix, matrix)
class SurfaceApproachSampler():
    """A grasp sampler that aligns the gripper's approach vector with the object surface.
    Args:
        gripper (graspsampling.hands.Hand): Gripper.
        object_mesh (trimesh.Trimesh): Object mesh.
        surface_normal_cone (float, optional): [description]. Defaults to 0.0.
        approach_cone (float, optional): [description]. Defaults to 0.0.
    """

    def __init__ (self, gripper, surface_normal_cone, approach_cone, normal_vector,neg_normal_vector,points):
        """Iniitialize attributes."""
        self.surface_normal_cone = surface_normal_cone * np.pi / 180
        self.approach_cone = approach_cone

        self.gripper = gripper
        #self.object_mesh = object_mesh
        self.mesh_normals=normal_vector
        self.mesh_points=points
        self.orignal=normal_vector[0]
        self.orignal_neg=neg_normal_vector
    def sample(self, number_of_grasps):
        """Sample grasp poses.
        Args:
            number_of_grasps (int): Number of grasps to sample.
        Raises:
            NotImplementedError: Some parameter values (approach cone > 0)
            are not yet implemented.
        Returns:
            dict: Contains grasp poses (7D: x, y, z, qx, qy, qz, qw).
        """
        #mesh_points, face_indices = self.object_mesh.sample(
            #number_of_grasps, return_index=True
        #)
        #mesh_normals = self.object_mesh.face_normals[face_indices]
        #self.orignal=self.mesh_normals
        self.mesh_normals= np.repeat(self.mesh_normals, repeats=number_of_grasps, axis=0)
        self.mesh_points= np.repeat(self.mesh_points, repeats=number_of_grasps, axis=0)

        if self.surface_normal_cone > 0.0:
            # perturb normals randomly
            normals = sample_spherical_cap(
                self.orignal,self.mesh_normals, self.surface_normal_cone, num_samples_per_dir=1
            )
        else:
            normals = self.mesh_normals

        # sample random standoffs
        logging.debug(
            "Standoff: %f, %f"
            % (self.gripper[0], self.gripper[1])
        )
        standoffs = np.random.uniform(
            self.gripper[0],
            self.gripper[1],
            size=(number_of_grasps, 1),
        )
        
        positions = self.mesh_points + normals * standoffs
        #print(self.mesh_points,normals,standoffs,positions)
        #print(self.orignal)
        surface_orientations = np.array(
            [trimesh.geometry.align_vectors([1,0,0], normal) for normal in normals]
        )

        roll_orientations = np.array(
            [
                tra.quaternion_about_axis(angle, [-1,0,0])
                for angle in np.random.uniform(0, 2*np.pi,size=(number_of_grasps, 1))
            ]
        )
        roll_orientations = np.array(
            [tra.quaternion_matrix(q) for q in roll_orientations]
        )
        if self.approach_cone > 0.0:
            raise NotImplementedError(
                "Feature is not implemented! (sampling in approach_cone > 0)"
            )

        orientations = np.array(
            [np.dot(s, r) for s, r in zip(surface_orientations, roll_orientations)]
        )
        orientations_q = np.array(
            [tra.quaternion_from_matrix(m, isprecise=True) for m in orientations]
        )

        poses = np.hstack([positions, orientations_q])

        return {"poses": poses.tolist()}   
class overlap_check:
    def __init__(self,extent,origin,rotation):

        self.origin = carb.Float3(origin[0], origin[1], origin[2])
        self.extent = carb.Float3(extent[0], extent[1], extent[2])
        self.rotation= carb.Float4(rotation[0],rotation[1],rotation[2],rotation[3])

    def report_hit(self,hit):
        return True

    def collision_check(self):
        # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
        # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
        #origin = carb.Float3(origin[0], origin[1], origin[2])
        #rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
        #self.distance = 10.0
        # physX query to detect closest hit
        numHits = get_physx_scene_query_interface().overlap_box( self.extent, self.origin, self.rotation, self.report_hit, False)
        return numHits>0
def draw_rotated_box(draw, origin, extent, rotation):
    # Build the rotation matrix
    #R = Rotation.from_quat(rotation)
    rotation_matrix =rotation
    #rotation_matrix[:3, :3] = rotation

    # Create the local corners of the box with respect to the origin
    local_corners = [
        np.array([origin[0] - extent[0], origin[1] - extent[1], origin[2] - extent[2], 1.0]),
        np.array([origin[0] + extent[0], origin[1] - extent[1], origin[2] - extent[2], 1.0]),
        np.array([origin[0] + extent[0], origin[1] + extent[1], origin[2] - extent[2], 1.0]),
        np.array([origin[0] - extent[0], origin[1] + extent[1], origin[2] - extent[2], 1.0]),
        np.array([origin[0] - extent[0], origin[1] - extent[1], origin[2] + extent[2], 1.0]),
        np.array([origin[0] + extent[0], origin[1] - extent[1], origin[2] + extent[2], 1.0]),
        np.array([origin[0] + extent[0], origin[1] + extent[1], origin[2] + extent[2], 1.0]),
        np.array([origin[0] - extent[0], origin[1] + extent[1], origin[2] + extent[2], 1.0]),
    ]

    # Transform the corners to world coordinates
   # finger_left_T = np.eye(4)
    


    world_corners = [np.matmul(rotation, corner) for corner in local_corners]

    # Convert to Cartesian coordinates
    world_corners = [corner[:3] / corner[3] for corner in world_corners]

    # Convert to carb Float3 format
    world_corners = [carb.Float3(corner[0], corner[1], corner[2]) for corner in world_corners]

    # Draw the lines that form the box using draw_lines
    color = carb.Float4(1,1,0,1)  # Yellow color
    draw.draw_lines([world_corners[0], world_corners[1], world_corners[4], world_corners[5]], 
                    [world_corners[1], world_corners[2], world_corners[5], world_corners[6]], 
                    [color, color, color, color], 
                    [1, 1, 1, 1])
                    
    draw.draw_lines([world_corners[2], world_corners[3], world_corners[6], world_corners[7]], 
                    [world_corners[3], world_corners[0], world_corners[7], world_corners[4]], 
                    [color, color, color, color], 
                    [1, 1, 1, 1])

    draw.draw_lines([world_corners[0], world_corners[1], world_corners[2], world_corners[3]], 
                    [world_corners[4], world_corners[5], world_corners[6], world_corners[7]], 
                    [color, color, color, color], 
                    [1, 1, 1, 1])
def sample_pinch_grasp (rotation_angle,suction_translation,suction_rotation_matrix,draw,the_world,visualize_flag):
            #score_grasp_on_suction_point=[]
            #for j in range(num_yaw_samples):
                    # Compute the rotation angle in radians
                    #rotation_angle = 2 * np.pi * j/ num_yaw_samples  # This will range from 0 to 2*pi

            # Rotate the gripper's orientation by this angle around the z-axis
            #rotated_orientation = rotate_matrix(orientation, rotation_angle)


            rotation_axis = suction_rotation_matrix[:, 0] 

            # Compute the rotation matrix
            
            # Apply the 90 degrees rotation to the suction rotation matrix
            #suction_rotation_matrix_new = np.matmul(rotation_matrix_y, suction_rotation_matrix)

            #suction_rotation_matrix_new = suction_rotation_matrix[[2, 1, 0], :]

            rotation_matrix = rotation_matrix_from_axis_and_angle(rotation_axis, rotation_angle)

            # Rotate the input matrix

            #print(suction_rotation_matrix)
            #print(rotation_matrix)
            suction_rotation_matrix = suction_rotation_matrix.reshape((3,3))
            rotation_matrix=rotation_matrix.reshape((3,3))

            suction_rotation_matrix = np.matmul(rotation_matrix, suction_rotation_matrix)

            #print(suction_rotation_matrix)

            suction_rotation_matrix_new = suction_rotation_matrix[:, [1, 2, 0]]
            suction_rotation_matrix_new[:, 0] = -suction_rotation_matrix_new[:, 0]
            
            
            #print(suction_rotation_matrix_new)
            ##########################################
        




            rotated_orientation_quat = euler_angles_to_quat(matrix_to_euler_angles(suction_rotation_matrix_new))

            # Add the rotated pose to the list
            #gripper_poses.append((suction_translation, rotated_orientation_quat))
            
            #add_reference_to_stage(get_assets_root_path(
            # ) + "/Isaac/Props/UIElements/frame_prim.usd", f"/target_{j}")
            #hand_list="/home/ubuntu/Documents/tulip_mesh.usd"
            # asset=create_prim(
                            #prim_path=f"/target_{j}", 
                            #prim_type="hand",
                            #scale = np.array([1,1,1]),
                            # usd_path=hand_list,
                            #position=suction_translation,
                            #translation=[-31.16753,-1.05492,3.51197],
                            # orientation=rotated_orientation_quat
                            # orientation=[0.6895,0.20083,-0.67799,0.16507],
                            #semantic_label=b
                        #)
                        #xform_prim = XFormPrim(asset.GetPath())

                        #xform_prim.set_world_pose(position =np.array([-31.16753,-1.05492,3.51197]),orientation=quaternion_wxyz)   

            #add_reference_to_stage(
            # "/home/ubuntu/Documents/tulip_mesh.usd", f"/target_{j}")

            #frame = XFormPrim(f"/target_{j}", scale=[1, 1, 1])
            # position = suction_translation
            #orientation = rotated_orientation_quat


                #check=overlap_check_mesh(hand_path[0],hand_path[1])

            Rq=suction_rotation_matrix_new
            #ox_base_origin_translation=[5.5+12/2,0,0]
            

            origin_T = np.eye(4)
            origin_T[:3, :3] = Rq
            origin_T[0:3, 3] = suction_translation
            #origin[1, 3] = suction_prim[1]
            #origin[2, 3] = suction_prim[2]


                
            box_base_T = np.eye(4)
            #T2[:3, :3] = R
            box_base_T[0:3, 3] = [0,0,11.5]
        
            
            box_base_T=np.matmul(origin_T, box_base_T)
                    

            finger_left_T = np.eye(4)
            #T2[:3, :3] = R (-1.4/2)-(10/2)
            finger_left_T[0:3, 3] = [(-1.5/2)-(10/2),0,5.5/2]
            finger_left_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            finger_left_extent=[1.5/2,2.5/2,7/2]

            finger_right_T = np.eye(4)
            #T2[:3, :3] = R
            finger_right_T[0:3, 3] =  [(1.5/2)+(10/2),0,5.5/2]
            finger_right_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            finger_right_extent=[1.5/2,2.5/2,7/2]

            box_middle_T = np.eye(4)
            #T2[:3, :3] = R
            box_middle_T[0:3, 3] = [0,0,5.5/2]
            box_middle_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            box_middle_extent=[0.5,0.5,0.5]




            finger_left_T=np.matmul(origin_T, finger_left_T)

            finger_right_T=np.matmul(origin_T, finger_right_T)

            box_middle_T=np.matmul(origin_T, box_middle_T)

            
            box_base_origin=box_base_T[0:3, 3]
            box_base_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            box_base_extent=[13/2,7.4/2,12/2]

            
            
            overlap_api1=overlap_check(box_base_extent,box_base_origin,box_base_rotation)
            overlap1=overlap_api1.collision_check()

            overlap_api2=overlap_check(finger_right_extent,finger_right_T[0:3, 3],finger_right_rotation)
            overlap2=overlap_api2.collision_check()

            overlap_api3=overlap_check(finger_left_extent,finger_left_T[0:3, 3],finger_left_rotation)
            overlap3=overlap_api3.collision_check()

            overlap_api4=overlap_check(box_middle_extent,box_middle_T[0:3, 3],box_middle_rotation)
            overlap4=overlap_api4.collision_check()
            overlap4=not overlap4
            #print(overlap1,overlap2,overlap3,overlap4)

            overlaps = [overlap1, overlap2, overlap3,overlap4]
    
            # Count the number of False occurrences
            false_count = overlaps.count(False)
            
            # Calculate the score
            collision_quality = false_count / len(overlaps)
            
           

            if collision_quality==1:
                if visualize_flag ==True:
                    draw_rotated_box(draw,[0,0,11.5],box_base_extent,origin_T)
                    draw_rotated_box(draw,[(-1.4/2)-(10/2),0,5.5/2],finger_left_extent,origin_T)
                    draw_rotated_box(draw,[(1.4/2)+(10/2),0,5.5/2],finger_right_extent,origin_T)
                    draw_rotated_box(draw,[0,0,5.5/2],box_middle_extent,origin_T)
                    points=[(suction_translation[0],suction_translation[1],suction_translation[2])]
                    draw.draw_points(points, [(1, 0, 0, 1)] * len(points), [10]*len(points))
                    simulation_app.update()
                    #for kkk in range(2):
                    the_world.step(render=True)  
                    draw.clear_lines()

        


            return suction_translation, suction_rotation_matrix_new, collision_quality
def compute_sampled_rotations(R, sampled_directions):
    """
    Compute the sampled rotation matrices given the original rotation matrix and
    sampled approach directions.

    Args:
    - R (np.ndarray): Original rotation matrix.
    - sampled_directions (np.ndarray): Array of sampled approach directions.

    Returns:
    - list: List of sampled rotation matrices.
    """
    original_normal = R[:, 0]  # Assuming the normal vector is the first column of R
    sampled_rotations = []

    for dir in sampled_directions:
        if np.allclose(original_normal, dir):
            # If the direction is the same as the original normal, no rotation is needed
            sampled_rotations.append(R)
        else:
            # Compute the rotation axis and angle
            axis = np.cross(original_normal, dir)
            theta = np.arccos(np.dot(original_normal, dir))
            
            # Compute the rotation matrix for aligning the direction with the original normal
            align_rotation = rodrigues_rotation_matrix(axis, theta)
            
            # Combine the alignment rotation with the original rotation matrix
            sampled_R = np.dot(align_rotation, R)
            sampled_rotations.append(sampled_R)

    return sampled_rotations
def rodrigues_rotation_matrix(axis, theta):
    """
    Compute the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians using the Rodrigues' rotation formula.

    Args:
    - axis (np.ndarray): Axis of rotation (should be a unit vector).
    - theta (float): Rotation angle in radians.

    Returns:
    - np.ndarray: Corresponding rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
def sample_cone(normal, num_samples, aperture_angle_deg):
    """
    Sample directions within a cone defined by a normal vector and an aperture angle.

    Args:
    - normal (np.ndarray): 3D normal vector defining the direction of the cone.
    - num_samples (int): Number of samples to generate.
    - aperture_angle_deg (float): Aperture angle of the cone in degrees.

    Returns:
    - np.ndarray: Array of sampled directions within the cone.
    """
    sampled_directions = []
    
    while len(sampled_directions) < num_samples:
        # Sample a random direction on the unit hemisphere
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        random_dir = np.array([x, y, z])
        
        # Check if the sampled direction is within the cone
        angle_rad = np.arccos(np.dot(random_dir, normal))
        if angle_rad <= np.radians(aperture_angle_deg):
            sampled_directions.append(random_dir)
            
    return np.array(sampled_directions)

class generate_pcl_and_seal():
    def __init__(self, data_root):
        self.data_root = data_root
        self.pointcloud_root= args.pcl_path
        self.stage_root = os.listdir(data_root)
        with open(args.seg_dic_path, "rb") as f:
                seg_dic=pickle.load(f)
        f.close()
        self.seg_dic=seg_dic


    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)
    
    def __getitem__(self, stage_ind,save_pcl_flag,save_seal_flag):
        # Load the stage asynchronously
        stage_folder=self.data_root+f"/stage_{stage_ind}"
        if args.instanceable_flag==True:
            stage_usd_list=self.data_root+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}_instanceable.usd"
        else:
            stage_usd_list=self.data_root+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}.usd"
        setup_task = asyncio.ensure_future(self.load_stage(stage_usd_list))

        while not setup_task.done():
            simulation_app.update()

        # Create World
        the_world = World(stage_units_in_meters = 0.01)
        
        if args.instanceable_flag==True:
            # Create a ground plane
            the_world.scene.add_ground_plane(size=1000, color=np.array([1, 1, 1]))

        stage = omni.usd.get_context().get_stage()
         
        pcd_annotators,sem_annots=get_camera_info()
            
        # Update and play the simulation

        # Remove rigid bodies and set colliders for all mesh primitives
        # Rigid body will use convex hull as collider default which is not accurate
 
        simulation_app.update()
        the_world.play()

        for i in range(150):
            the_world.step(render=True)

        curr_prim = stage.GetPrimAtPath("/World/objects")
        for prim in Usd.PrimRange(curr_prim):
            if prim.IsA(UsdGeom.Xform):
                utils.removeRigidBody(prim)
            elif prim.IsA(UsdGeom.Mesh):
                utils.setCollider(prim, approximationShape="convexDecomposition")

     
        all_unique_classes, label_to_object_number_map=get_semantic_info(pcd_annotators,sem_annots)
    
        # For each unique class, combine point clouds from all cameras, and perform raycasting and collision detection
        point_cloud_dic,all_instance_seg_ids=get_merge_pointcloud(all_unique_classes,pcd_annotators,label_to_object_number_map,stage_folder,args.save_pcd_flag)
       
        draw = _debug_draw.acquire_debug_draw_interface()

        draw.clear_points()
        draw.clear_lines()
        
        # Create a map for object number to label
        object_number_to_label_map = {v: k for k, v in label_to_object_number_map.items()}
       
        # Initial lists and dictionaries
        display=[]
        candidates={}

        # Placeholder array for storing point clouds
        point_clouds_stage=np.empty((0,10), float)

        # Iterate over all unique segment ids
        print(all_instance_seg_ids)

        for obj_num in all_instance_seg_ids:
            candidates[obj_num]=dict()
            candidates[obj_num]["grasp_samples"] = []
            class_name=object_number_to_label_map[obj_num]
            candidates[obj_num]["segmentation_id"]= self.seg_dic[class_name]
            candidates[obj_num]["object_name"]=class_name
            print(object_number_to_label_map[obj_num])
            # If current instance is ground
            if obj_num==0:
                
                pcd=point_cloud_dic[obj_num]
                display.append(pcd)
                points=np.asarray(pcd.points)
                normals=np.asarray(pcd.normals)
                segmentation_ground=np.full(shape=points.shape[0],fill_value=(0),dtype=np.int32)
                segmentation=segmentation_ground[...,np.newaxis]
            
            # For all other instances (non-ground)
            if obj_num!=0:
                pcd=point_cloud_dic[obj_num]
                segmentation=np.full(shape=np.asarray(pcd.points).shape[0],fill_value=((self.seg_dic[class_name])),dtype=np.int32)
                segmentation = segmentation[..., np.newaxis]
                # Convert points to tensor for sampling
                torch_pts=torch.from_numpy(np.asarray(pcd.points))
                torch_pts = torch.unsqueeze(torch_pts, dim=0)
                if torch_pts.shape[1]<500:
                    point_idx = farthest_point_sampler(torch_pts, torch_pts.shape[1])
                else:  
                    point_idx = farthest_point_sampler(torch_pts, 500)

                np_arr = point_idx.cpu().detach().numpy()
                FPS_index=np.squeeze(np_arr)
             
                kdtree = o3d.geometry.KDTreeFlann(pcd)
              
                points = np.asarray(pcd.points) # Nx3 np array of points
                normals = np.asarray(pcd.normals) # Nx3 np array of normals
                
                # Ensure points and normals have the same length
                if len(normals)!= len(points):
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.suction_radius, max_nn=500))
                    normals = np.asarray(pcd.normals)

            
                # For each index in the index list
                np.random.shuffle(FPS_index)

                for index in FPS_index:
                    #####################compute_darboux_frame###################################################################
                    #print(index)
                    t_ori, t_translate, R,normal_vector,normal_vector_opposite=compute_darboux_frame(points, index, kdtree, normals)
                    # Sample approach directions within a cone around the normal vector

                    sampled_approach_directions = sample_cone(normal_vector, 3, 10)

                    # Compute the corresponding rotation matrices for each sampled direction
                    sampled_rotations = compute_sampled_rotations(R, sampled_approach_directions)
                    #print(sampled_rotations)
                    

                    #standoff_depth=[0,0]
                  
                    #####################collision check###################################################################

                    if (check_raycast(t_translate,normal_vector))== None:
                        collision_flag=False
                    else:
                        collision_flag=True
        
                            
                   
                    ###############################################################################
                    
                    standoff_depths = np.linspace(1, 7, num=7) 


                    if collision_flag== False:
                        #print("collision pass")
                        #####################seal evaluation###################################################################
                        pinch_grasp_samples=12
                        # Create a gripper object
                        for sampled_rotations_j in range(len(sampled_rotations)): 
                            for pinch_grasp_samples_j in range(pinch_grasp_samples):
                                for standoff_j in standoff_depths:
                        # Compute the rotation angle in radians
                                    rotation_angle = 2 * np.pi * pinch_grasp_samples_j/ pinch_grasp_samples  # This will range from 0 to 2*pi
                                
                                    offset_translation = t_ori - standoff_j * sampled_approach_directions[sampled_rotations_j]  # Note the minus sign


                                    grasp_translation, grasp_rot, collision_quality=sample_pinch_grasp(rotation_angle,offset_translation,sampled_rotations[sampled_rotations_j],draw,the_world,False)                                     
                        
                                    
                                    candidates[obj_num]["grasp_samples"].append({
                                        "segmentation_id":self.seg_dic[class_name],
                                        "object_name": class_name,
                                        "t_ori":t_ori,
                                        "approach_direction": sampled_approach_directions[sampled_rotations_j],
                                        "suction_translation": grasp_translation,
                                        "suction_rotation_matrix": grasp_rot,
                                        "stand_off":standoff_j,
                                        "collision_quality": collision_quality,
                                    })


                    ##############################################isaac_visual######################################
                    passed_samples_count = len([sample for sample in candidates[obj_num]["grasp_samples"] if sample["collision_quality"] == 1])
                    if passed_samples_count >= 300:
                        break

                    ##################################################################################################
                if save_seal_flag == True:
                    # Separate out good and bad candidates
                    good_candidates = [sample for sample in candidates[obj_num]["grasp_samples"] if sample["collision_quality"] == 1]
                    bad_candidates = [sample for sample in candidates[obj_num]["grasp_samples"] if sample["collision_quality"] != 1]

                    # Randomly select up to 500 of the bad candidates
                    selected_bad_candidates = random.sample(bad_candidates, min(500, len(bad_candidates)))

                    # Combine good candidates with selected bad ones
                    candidates[obj_num]["grasp_samples"] = good_candidates + selected_bad_candidates

                    with open(self.data_root+f"/stage_{stage_ind}/"+f"stage_{stage_ind}_grasp_candidates_after_overlap"+".pkl", "wb") as f:
                        pickle.dump(candidates, f)
                    f.close()

            color = np.asarray(pcd.colors)  # assuming color is another attribute of your pcd object

            points_10 = np.hstack([points, normals,color,segmentation]) #(N,10)

            point_clouds_stage=np.append(point_clouds_stage, points_10, axis=0)

        if save_pcl_flag == True:
            np.savez_compressed(self.pointcloud_root+f"/{stage_ind}.npz",point_clouds_stage)
        simulation_app.update()
 
        return None

    def __len__(self):
        return len(self.stage_root)


data_root=args.data_path
stage_root = os.listdir(data_root)
my_dataset = generate_pcl_and_seal(data_root=data_root)

for stage_ind in range(args.start_stage,args.end_stage):
    
    my_dataset.__getitem__(stage_ind,args.save_pcl_flag,args.save_pkl_flag)













