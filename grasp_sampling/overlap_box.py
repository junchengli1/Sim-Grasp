from omni.isaac.kit import SimulationApp
simulation_app=SimulationApp({"headless": False})
from omni.isaac.core import World
from pxr import Gf, Sdf, UsdGeom, UsdShade, Semantics, UsdPhysics,PhysicsSchemaTools
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.range_sensor import _range_sensor 
from omni.syntheticdata import _syntheticdata
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
import omni
import carb
import numpy as np
import os
import glob
import pickle
import json 
import random
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_rot_matrix,quat_to_euler_angles
from omni.isaac.universal_robots import UR10
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.prims import XFormPrim, RigidPrim
import logging
from omni.isaac.dynamic_control import _dynamic_control
#from tf.transformations import quaternion_matrix
import time
import copy
import matplotlib.pyplot as plt
from einops import rearrange
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.syntheticdata import visualize
import open3d as o3d

from omni.isaac.debug_draw import _debug_draw

from typing import Optional

import os
import torch
import torch.utils.data
import torchvision
import pickle
from omni.isaac.core.utils.viewports import set_camera_view

import math as m
import omni.physx
from omni.physx import get_physx_scene_query_interface

from pxr import UsdGeom
from pxr import Vt,Gf
from collections import deque
import math
from PIL import Image
from pxr import UsdGeom

import omni.replicator.core as rep
from omni.replicator.core import Writer, AnnotatorRegistry
import trimesh
import trimesh.transformations as tra

voxel_size = 0.02

max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

import math
import typing
import numpy as np

# omniverse
from pxr import Gf
from omni.syntheticdata import helpers


def euler_to_rot_matrix(euler_angles: np.ndarray, degrees: bool = False) -> Gf.Rotation:
    """Convert from Euler XYZ angles to rotation matrix.

    Args:
        euler_angles (np.ndarray): Euler XYZ angles.
        degrees (bool, optional): Whether input angles are in degrees. Defaults to False.

    Returns:
        Gf.Rotation: Pxr rotation object.
    """
    return Gf.Rotation(Gf.Quatf(*euler_angles_to_quat(euler_angles, degrees)))

def get_circle_coord(theta, x_center, y_center, radius):
    x = 0
    y = radius * math.sin(theta) + y_center
    z=radius * math.cos(theta) + x_center
    return (x,y,z)

# This function gets all the pairs of coordinates
def get_all_circle_coords(x_center, y_center, radius, n_points):
    thetas = [i/n_points * math.tau for i in range(n_points)]
    circle_coords = [get_circle_coord(theta, x_center, y_center, radius) for theta in thetas]
    return circle_coords

# Using the second function to generate all the pairs of coordinates.


class FPS:
    def __init__(self, pcd_xyz, n_samples):
        self.n_samples = n_samples
        self.pcd_xyz = pcd_xyz
        self.n_pts = pcd_xyz.shape[0]
        self.dim = pcd_xyz.shape[1]
        self.selected_pts = None
        self.selected_pts_expanded = np.zeros(shape=(n_samples, 1, self.dim))
        self.remaining_pts = np.copy(pcd_xyz)
        self.grouping_radius = None
        self.dist_pts_to_selected = None  # Iteratively updated in step(). Finally re-used in group()
        self.labels = None
        # Random pick a start
        self.start_idx = np.random.randint(low=0, high=self.n_pts - 1)
        self.selected_pts_expanded[0] = self.remaining_pts[self.start_idx]
        self.n_selected_pts = 1
        self.dist_pts_to_selected_min = None
        self.res_selected_idx=None
        self.index=[]
    def get_selected_pts(self):
        self.selected_pts = np.squeeze(self.selected_pts_expanded, axis=1)
        return self.selected_pts

    def step(self):
        #print(self.n_selected_pts)
        if self.n_selected_pts == 1:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts, self.selected_pts_expanded[:self.n_selected_pts]).T
            self.dist_pts_to_selected_min = np.min(self.dist_pts_to_selected, axis=1, keepdims=True)
            self.res_selected_idx = np.argmax(self.dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[self.res_selected_idx]
            #print(self.res_selected_idx)
            self.n_selected_pts += 1


        elif self.n_selected_pts < self.n_samples:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts, np.expand_dims(np.expand_dims(self.remaining_pts[self.res_selected_idx],0),0)).T 
            for i in range(0,self.remaining_pts.shape[0]):
                if self.dist_pts_to_selected_min[i]>self.dist_pts_to_selected[i]:
                    self.dist_pts_to_selected_min[i]=self.dist_pts_to_selected[i]
            self.res_selected_idx = np.argmax(self.dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[self.res_selected_idx]
            #print(self.res_selected_idx)
            self.n_selected_pts += 1
        else:
            print("Got enough number samples")


    def fit(self):
        
        for _ in range(1, self.n_samples):
            self.step()
            self.index.append(self.res_selected_idx)
            #print(self.index)
            #print("sampleing no.",_," point")
            #print(self.get_selected_pts())
        return self.index

    def group(self, radius):
        self.grouping_radius = radius   # the grouping radius is not actually used
        dists = self.dist_pts_to_selected

        # Ignore the "points"-"selected" relations if it's larger than the radius
        dists = np.where(dists > radius, dists+1000000*radius, dists)

        # Find the relation with the smallest distance.
        # NOTE: the smallest distance may still larger than the radius.
        self.labels = np.argmin(dists, axis=1)
        return self.labels


    @staticmethod
    def __distance__(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)



def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance_coarse, np.identity(4),o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance_fine,icp_coarse.transformation,o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, max_correspondence_distance_fine,icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,target_id,transformation_icp,information_icp, uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id, target_id,transformation_icp,information_icp, uncertain=True))
    return pose_graph


def distance_point3d(p0, p1):
    d = (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p0[2])**2
    return math.sqrt(d)

def furthest_point_sample(points, sample_count):
    points_index = np.arange(points.shape[0], dtype=np.int)
    A = np.array([np.random.choice(points_index)])
    B = np.setdiff1d(points_index, A)
    print(A)
    print(B)
    min_dis_B2A = []
    for i in range(len(B)):
        Pa_index = A[0]
        Pb_index = B[i]
        Pa = points[Pa_index]
        Pb = points[Pb_index]
        dis = distance_point3d(Pb, Pa)
        min_dis_B2A.append(dis)
    min_dis_B2A = np.array(min_dis_B2A)
    print('iter ', len(A), ': ', A)
    while len(A) < sample_count:
        longest_points_in_B_index = np.argmax(min_dis_B2A)
        longest_points_index = B[longest_points_in_B_index]

        # update A and B
        A = np.append(A, longest_points_index)
        B = np.delete(B, longest_points_in_B_index)
        min_dis_B2A = np.delete(min_dis_B2A, longest_points_in_B_index)

        # update min_dis_B2A
        for i in range(len(B)):
            Pa_index = A[-1]
            Pb_index = B[i]
            Pa = points[Pa_index]
            Pb = points[Pb_index]
            dis = distance_point3d(Pb, Pa)
            min_dis_B2A[i] = min(dis, min_dis_B2A[i])
        
        print('iter ', len(A), ': ', A)

    return A

class RenderClass(object):
    def __init__(self, pcd, points, sampled_index):
        self.pcd = pcd
        self.points = points
        self.sampled_index = sampled_index
        self.current_render_index = 0
    
    def vis_callback(self, vis):
        if self.current_render_index < len(self.sampled_index):
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            mesh_sphere.paint_uniform_color([0.8, 0.1, 0.2])
            mesh_sphere.translate(self.points[self.sampled_index[self.current_render_index]])
            self.current_render_index = self.current_render_index + 1
            time.sleep(0.1)
            vis.add_geometry(mesh_sphere, False)
        else:
            vis.clear_geometries()
            vis.add_geometry(self.pcd, False)
            self.current_render_index = 0



def get_meters_per_unit():
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    return UsdGeom.GetStageMetersPerUnit(stage)

def gf_as_numpy(gf_matrix)->np.array:
    """Take in a pxr.Gf matrix and returns it as a numpy array.
    Specifically it transposes the matrix so that it follows numpy
    matrix rules.

    Args:
        gf_matrix (Gf.Matrix_d): Gf matrix to convert

    Returns:
        np.array:
    """
    # Convert matrix/vector to numpy array
    return np.array(list(gf_matrix)).T

def get_intrinsic_matrix(viewport,sd_helper):
    # Get camera params from usd
    
    camera=sd_helper.get_camera_params(viewport)
    focal_length = camera['focal_length']
    horiz_aperture= camera['horizontal_aperture']
    width=camera['resolution']['width']
    height=camera['resolution']['height']
    
    
   # stage = omni.usd.get_context().get_stage()
    #prim = stage.GetPrimAtPath(viewport.get_active_camera())
    #focal_length = prim.GetAttribute("focalLength").Get()
    #horiz_aperture = prim.GetAttribute("horizontalAperture").Get()
    #x_min, y_min, x_max, y_max = viewport.get_viewport_rect()
   # width, height = x_max - x_min, y_max - y_min
    
    # Pixels are square so we can do:
    vert_aperture = height / width * horiz_aperture

    # Compute focal point and center
    focal_x = width * focal_length / horiz_aperture
    focal_y = height * focal_length / vert_aperture
    center_x = width * 0.5
    center_y = height * 0.5
    
    # Turn into matrix
    intrinsic_matrix = np.array([[focal_x, 0, center_x],
                                 [0, focal_y, center_y],
                                 [0, 0, 1]])
    print(intrinsic_matrix)
    return intrinsic_matrix

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    '''
    
    **Input:**
    - center: numpy array of (3,), target point as gripper center
    - R: numpy array of (3,3), rotation matrix of gripper
    - width: float, gripper width
    - score: float, grasp quality score
    **Output:**
    - open3d.geometry.TriangleMesh
    '''
    x, y, z = center[0],center[1],center[2]
    height=0.004*100
    finger_width = 0.004*100
    tail_length = 0.04*100
    depth_base = 0.02*100
    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score # red for high score
        color_g = 0
        color_b = 1 - score # blue for low score
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def get_extrinsic_matrix(viewport,sd_helper, meters=False):
    camera=sd_helper.get_camera_params(viewport)
    #print(camera)
    pose = camera['pose']
    camera_pose=pose.T
    #print(camera)
    #from pxr import UsdGeom
    # Get camera pose
    #stage = omni.usd.get_context().get_stage()
    #camera_prim = stage.GetPrimAtPath(viewport.get_active_camera())
    #print(UsdGeom.Camera(camera_prim).GetLocalTransformation())
    #camera_pose = gf_as_numpy(UsdGeom.Camera(camera_prim).GetLocalTransformation())
    
    
    if meters:
        #print(camera_pose[:,3])
        camera_pose[:,3] = camera_pose[:,3]*get_meters_per_unit()
        #camera_pose[:,3][3]=1
        
    #print(pose[:,3])
    view_matrix = np.linalg.inv(camera_pose)
    #print(view_matrix)
    return view_matrix

def freq_count(v:np.array)->np.array:
    """Return the number of times each element in an array occur

    Args:
        v (np.array): 1D array to count

    Returns:
        np.array: Frequency list [[num, count], [num, count],...]
    """
    unique, counts = np.unique(v, return_counts=True)
    return np.asarray((unique, counts)).T

def pointcloud_from_mask_and_depth(depth:np.array, mask:np.array, mask_val:int, intrinsic_matrix:np.array, extrinsic_matrix:np.array=None):
    depth = np.array(depth).squeeze()
    mask = np.array(mask).squeeze()
    # Mask the depth array
    masked_depth = np.ma.masked_where(mask!=mask_val, depth)
    masked_depth = np.ma.masked_greater(masked_depth, 8000)
    
    #plt.imshow(masked_depth)

    # Create idx array
    idxs = np.indices(masked_depth.shape)
    u_idxs = idxs[1]
    v_idxs = idxs[0]
    # Get only non-masked depth and idxs
    z = masked_depth[~masked_depth.mask]
    compressed_u_idxs = u_idxs[~masked_depth.mask]
    compressed_v_idxs = v_idxs[~masked_depth.mask]
    # Calculate local position of each point
    # Apply vectorized math to depth using compressed arrays
    cx = intrinsic_matrix[0,2]
    fx = intrinsic_matrix[0,0]
    cy = intrinsic_matrix[1,2]
    fy = intrinsic_matrix[1,1]
   

    x = (compressed_u_idxs - cx) * z / fx
    
    
    #cy = intrinsic_matrix[1,2]
    #fy = intrinsic_matrix[1,1]
    # Flip y as we want +y pointing up not down
    y = -(compressed_v_idxs - cy) * z / fy
    #print(z)
    # Apply camera_matrix to pointcloud as to get the pointcloud in world coords
    if extrinsic_matrix is not None:
        # Calculate camera pose from extrinsic matrix
        camera_matrix = np.linalg.inv(extrinsic_matrix)
        #print(camera_matrix)
        #print(camera_matrix.shape)
        #print(camera_matrix)
        # Create homogenous array of vectors by adding 4th entry of 1
        # At the same time flip z as for eye space the camera is looking down the -z axis
        w = np.ones(z.shape)
        x_y_z_eye_hom = np.vstack((x, y, -z, w))
        #print(x_y_z_eye_hom.T)
        # Transform the points from eye space to world space
        x_y_z_world = np.dot(camera_matrix, x_y_z_eye_hom)[:3]
        
        #x_y_z_world1 = np.dot(camera_matrix, x_y_z_eye_hom)[:4]
        #print(x_y_z_world1.T)
        return x_y_z_world.T
    else:
        x_y_z_local = np.vstack((x, y, z))
        return x_y_z_local.T
def check_raycast(origin,rayDir):
    # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
    # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
    origin = carb.Float3(origin[0], origin[1], origin[2])
    rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
    distance = 100.0
    # physX query to detect closest hit
    hit = get_physx_scene_query_interface().raycast_closest(origin, rayDir, distance,True)
    if(hit["hit"]):
        stage = omni.usd.get_context().get_stage()
        # Change object colour to yellow and record distance from origin
        usdGeom = UsdGeom.Mesh.Get(stage, hit["rigidBody"])
        hitColor = Vt.Vec3fArray([Gf.Vec3f(255.0 / 255.0, 255.0 / 255.0, 0.0)])
        #usdGeom.GetDisplayColorAttr().Set(hitColor)
        distance = hit["distance"]
        position=hit["position"]
        #print(hit)
       
        return usdGeom.GetPath().pathString, distance, position
    return None
class all_hits:
    def __init__(self,origin,rayDir):
        self.origin = carb.Float3(origin[0], origin[1], origin[2])
        self.rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
        #self.n_pts = pcd_xyz.shape[0]
        self.hit_position=[]
        self.hit_distance=[]
        self.hit_usd=[]
        self.distance=100.0
    def report_allhits(self,hit):
        stage = omni.usd.get_context().get_stage()
        usdGeom = UsdGeom.Mesh.Get(stage, hit.rigid_body)
        self.hit_usd.append(usdGeom.GetPath().pathString)
        #distance = hit["distance"]
        #position=hit["position"]
        #print(hit.face_index)
        self.hit_position.append([hit.position[0],hit.position[1],hit.position[2]])
        self.hit_distance.append(hit.distance)
        #print( self.hit_position)
        return True
    def check_raycast2(self):
        # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
        # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
        #origin = carb.Float3(origin[0], origin[1], origin[2])
        #rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
        #self.distance = 10.0
        # physX query to detect closest hit
        hit = get_physx_scene_query_interface().raycast_all(self.origin, self.rayDir,self.distance,self.report_allhits,bothSides=True)
        if(hit):
            #stage = omni.usd.get_context().get_stage()
            # Change object colour to yellow and record distance from origin
            #usdGeom.GetDisplayColorAttr().Set(hitColor)
            return self.hit_usd,self.hit_distance,self.hit_position

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


class overlap_check_mesh:
    def __init__(self,hand_path_0,hand_path_1):
        self.hand_path_0=hand_path_0
        self.hand_path_1=hand_path_1
        self.hit_usd=[]
    def report_hit(self,hit):
        stage = omni.usd.get_context().get_stage()
        usdGeom = UsdGeom.Mesh.Get(stage, hit.rigid_body)
        self.hit_usd.append(usdGeom.GetPath().pathString)
        #distance = hit["distance"]
        #position=hit["position"]
        #print(hit.face_index)
        #self.hit_position.append([hit.position[0],hit.position[1],hit.position[2]])

        #self.hit_distance.append(hit.distance)
        return True

    def collision_check(self):
        # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
        # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
        #origin = carb.Float3(origin[0], origin[1], origin[2])
        #rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
        #self.distance = 10.0
        # physX query to detect closest hit
        numHits = get_physx_scene_query_interface().overlap_mesh(self.hand_path_0,self.hand_path_1,self.report_hit, False)
        return self.hit_usd, numHits




def create_mesh_cylinder(R, t, radius=0.1, height=1):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    cylinder.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    print(cylinder.vertices)
    colors = np.array([0, 0, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)
    return cylinder

def create_mesh_cylinder_detection(R, t, collision, radius=0.5, height=1):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]

    #print(np.array(vertices))

    vertices[:, 0] += height / 2
 

    vertices = np.dot(R, vertices.T).T + t
    

    cylinder.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    if collision:
        colors = np.array([0, 0, 0])
    else:
        colors = np.array([1, 1, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder

def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
import asyncio
from omni.isaac.core.utils.stage import is_stage_loading

# Save rgb image to file
def save_rgb(rgb_data, file_name):
    rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
    rgb_img = Image.fromarray(rgb_image_data, "RGBA")
    rgb_img.save(file_name + ".png")


# Randomize cube color every frame using a replicator randomizer
def cube_color_randomizer():
    cube_prims = rep.get.prims(path_pattern="Cube")
    with cube_prims:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
    return cube_prims.node

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

class Dataset():
    def __init__(self, data_root):
        self.data_root = data_root
        self.stage_root = os.listdir(data_root)
        #print(self.stage_root) 
        #self.kit = SimulationApp(RENDER_CONFIG)
        #self.simulation_world = World(stage_units_in_meters = 0.01)
        global viewport_window
        global viewport_window1
        global viewport_window2
        global viewport_window3
        #global viewport_interface
       
        #viewport_window=None
        #viewport_window1=None
        #viewport_window2=None

    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)
    
    def make_plots(self, gt, name_suffix=""):
        _, axes = plt.subplots(1, 3, figsize=(40, 14))
        axes = axes.flat
        for ax in axes:
            ax.axis("off")
        # RGB
        axes[0].set_title("RGB")
        for ax in axes[:-1]:
            ax.imshow(gt["rgb"])

        # DEPTH
        axes[1].set_title("Depth")
        depth_data = np.clip(gt["depth"], 0, 255)
        axes[1].imshow(visualize.colorize_depth(depth_data.squeeze()))

        # SEMANTIC SEGMENTATION
        axes[2].set_title("Semantic Segmentation")
        semantic_seg = gt["semanticSegmentation"]
        #print(semantic_seg.shape)
        semantic_rgb = visualize.colorize_segmentation(semantic_seg)
        axes[2].imshow(semantic_rgb)

        # Save figure
        print("saving figure to: ", name_suffix)
        plt.savefig(name_suffix)
        
    def save_point_cloud(self,stage_name,folder_name,info,the_world,draw,semantic_seg_ids,semantic_seg,depth_linear,intrinsic_matrix,extrinsic_matrix,camera_location):
        point_cloud_dic={}
        point_cloud={}
        for i in semantic_seg_ids:
            if i!=0:
                print(info[i-1][3])

                points = pointcloud_from_mask_and_depth(depth_linear, semantic_seg, i, intrinsic_matrix, extrinsic_matrix)
                points = (1.0/get_meters_per_unit()) * points
                point_cloud_dic[i] = o3d.geometry.PointCloud()
                point_cloud_dic[i].points = o3d.utility.Vector3dVector(points)
               # point_cloud_dic[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30),fast_normal_computation=False)
                point_cloud_dic[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParam(radius=5,fast_normal_computation=False))

                point_cloud_dic[i].orient_normals_towards_camera_location(camera_location=camera_location)

                r,g,b=random.random(),random.random(),random.random()
                point_cloud_dic[i].paint_uniform_color([r, g, b])
                
                #point_cloud_dic[i],ind = point_cloud_dic[i].remove_radius_outlier(nb_points=150, radius=5)

                point_cloud_dic[i],ind = point_cloud_dic[i].remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
                o3d.io.write_point_cloud(folder_name+info[i-1][3]+"_"+str(info[i-1][0])+".pcd",point_cloud_dic[i])
                point_cloud[i]=point_cloud_dic[i].points
            # Draw lines
                draw.draw_points(points, [(r, g, b, 1)] * len(points), [1]*len(points))
                simulation_app.update()
                the_world.step()
                #the_world.render()
        
        return point_cloud_dic

    def combine(self,stage_name,folder_name,info,the_world,draw,instance_seg_ids,instance_seg,depth_linear,intrinsic_matrix,extrinsic_matrix,camera_location,info1,instance_seg_ids1,instance_seg1,depth_linear1,intrinsic_matrix1,extrinsic_matrix1,camera_location1,info2,instance_seg_ids2,instance_seg2,depth_linear2,intrinsic_matrix2,extrinsic_matrix2,camera_location2,info3,instance_seg_ids3,instance_seg3,depth_linear3,intrinsic_matrix3,extrinsic_matrix3,camera_location3,info4,instance_seg_ids4,instance_seg4,depth_linear4,intrinsic_matrix4,extrinsic_matrix4,camera_location4):
        point_cloud_dic={}
        point_cloud_dic1={}
        point_cloud_dic2={}
        point_cloud_dic3={}
        point_cloud_dic4={}

        point_cloud={}
        all_instance_seg_ids= np.unique(np.concatenate((instance_seg_ids, instance_seg_ids1,instance_seg_ids2,instance_seg_ids3,instance_seg_ids4)))
        for i in all_instance_seg_ids:
            if i!=0:
                #print(info[i-1][3])
                if i in instance_seg_ids:
                    points = pointcloud_from_mask_and_depth(depth_linear, instance_seg, i, intrinsic_matrix, extrinsic_matrix)
                    points = (1.0/get_meters_per_unit()) * points
                    point_cloud_dic[i] = o3d.geometry.PointCloud()
                    point_cloud_dic[i].points = o3d.utility.Vector3dVector(points)
                    point_cloud_dic[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1),fast_normal_computation=False)
                    #point_cloud_dic[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=300),fast_normal_computation=False)
                    point_cloud_dic[i].orient_normals_towards_camera_location(camera_location=camera_location)
                else:
                    points=np.empty([1, 3])
                    point_cloud_dic[i] = o3d.geometry.PointCloud()
                    point_cloud_dic[i].points = o3d.utility.Vector3dVector(points)


                if i in instance_seg_ids1:
                    points1 = pointcloud_from_mask_and_depth(depth_linear1, instance_seg1, i, intrinsic_matrix1, extrinsic_matrix1)
                    points1 = (1.0/get_meters_per_unit()) * points1
                    point_cloud_dic1[i] = o3d.geometry.PointCloud()
                    point_cloud_dic1[i].points = o3d.utility.Vector3dVector(points1)
                    #point_cloud_dic1[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=300),fast_normal_computation=False)
                    point_cloud_dic1[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1),fast_normal_computation=False)
                    point_cloud_dic1[i].orient_normals_towards_camera_location(camera_location=camera_location1)
                else:
                    points1=np.empty([1, 3])
                    point_cloud_dic1[i] = o3d.geometry.PointCloud()
                    point_cloud_dic1[i].points = o3d.utility.Vector3dVector(points1)

                if i in instance_seg_ids2:
                    points2 = pointcloud_from_mask_and_depth(depth_linear2, instance_seg2, i, intrinsic_matrix2, extrinsic_matrix2)
                    points2 = (1.0/get_meters_per_unit()) * points2
                    point_cloud_dic2[i] = o3d.geometry.PointCloud()
                    point_cloud_dic2[i].points = o3d.utility.Vector3dVector(np.array(points2))
                   # point_cloud_dic2[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=300),fast_normal_computation=False)
                    point_cloud_dic2[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1),fast_normal_computation=False)
                    point_cloud_dic2[i].orient_normals_towards_camera_location(camera_location=camera_location2)
                else:
                    points2=np.empty([1, 3])
                    point_cloud_dic2[i] = o3d.geometry.PointCloud()
                    point_cloud_dic2[i].points = o3d.utility.Vector3dVector(points2)

                if i in instance_seg_ids3:
                    points3 = pointcloud_from_mask_and_depth(depth_linear3, instance_seg3, i, intrinsic_matrix3, extrinsic_matrix3)
                    points3 = (1.0/get_meters_per_unit()) * points3
                    point_cloud_dic3[i] = o3d.geometry.PointCloud()
                    point_cloud_dic3[i].points = o3d.utility.Vector3dVector(np.array(points3))
                   # point_cloud_dic2[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=300),fast_normal_computation=False)
                    point_cloud_dic3[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1),fast_normal_computation=False)
                    point_cloud_dic3[i].orient_normals_towards_camera_location(camera_location=camera_location3)
                else:
                    points3=np.empty([1, 3])
                    point_cloud_dic3[i] = o3d.geometry.PointCloud()
                    point_cloud_dic3[i].points = o3d.utility.Vector3dVector(points3)

                if i in instance_seg_ids4:
                    points4 = pointcloud_from_mask_and_depth(depth_linear4, instance_seg4, i, intrinsic_matrix4, extrinsic_matrix4)
                    points4 = (1.0/get_meters_per_unit()) * points4
                    point_cloud_dic4[i] = o3d.geometry.PointCloud()
                    point_cloud_dic4[i].points = o3d.utility.Vector3dVector(np.array(points4))
                   # point_cloud_dic2[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=300),fast_normal_computation=False)
                    point_cloud_dic4[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1),fast_normal_computation=False)
                    point_cloud_dic4[i].orient_normals_towards_camera_location(camera_location=camera_location4)
                else:
                    points4=np.empty([1, 3])
                    point_cloud_dic4[i] = o3d.geometry.PointCloud()
                    point_cloud_dic4[i].points = o3d.utility.Vector3dVector(points4)

                point_cloud[i]=point_cloud_dic[i]+point_cloud_dic1[i]+point_cloud_dic2[i]+point_cloud_dic3[i]+point_cloud_dic4[i]

                r,g,b=random.random(),random.random(),random.random()
                point_cloud[i].paint_uniform_color([r, g, b])
                
                #point_cloud_dic[i],ind = point_cloud_dic[i].remove_radius_outlier(nb_points=150, radius=5)

                point_cloud[i],ind = point_cloud[i].remove_statistical_outlier(nb_neighbors=20,std_ratio=2)
                #o3d.io.write_point_cloud(folder_name+info[i-1][3]+"_"+str(info[i-1][0])+".pcd",point_cloud[i])
                #point_cloud[i]=point_cloud_dic[i].points
            # Draw lines
                #draw.draw_points(points, [(r, g, b, 1)] * len(points), [1]*len(points))
                simulation_app.update()
                the_world.step()
                #the_world.render()
        
        return point_cloud

    def __getitem__(self, index):
        
        stage_folder=self.stage_root[index]
        print(stage_folder)
        usd_list=self.data_root+'/'+stage_folder+"/"+stage_folder+".usd"
        pkl_list=self.data_root+'/'+stage_folder+"/"+stage_folder+".pkl"

        with open(pkl_list, 'rb') as f:
            stage_info = pickle.load(f)
        #print(stage_info.keys())
        setup_task = asyncio.ensure_future(self.load_stage(usd_list))
        #simulation_world=World(stage_units_in_meters = 0.01)
        while not setup_task.done():
            simulation_app.update()
            #simulation_world.step()


        the_world = World(stage_units_in_meters = 0.01)
        #omni.isaac.core.utils.viewports.set_camera_view(eye=np.array([0, 0, 140]), target=np.array([0, 0, 0]))
        #set_camera_view(eye=np.array([0, 0, 140]), target=np.array([0, 0, 0]))
        stage = omni.usd.get_context().get_stage()
        #camera_path = "/World/Camera"
        #camera_prim = stage.DefinePrim(camera_path, "Camera")
        # Set as current camera
        
        #viewport_interface=None
        
        viewport_interface = omni.kit.viewport_legacy.get_viewport_interface()

        #viewport = viewport_interface.get_viewport_window()
        
        viewport_window=None
        viewport_window1=None
        viewport_window2=None
        viewport_window3=None
        viewport_window4=None
        #viewport_handle=None
        #viewport_handle1=None
        #viewport_handle2=None
        global viewport_handle
        global viewport_handle1
        global viewport_handle2
        global viewport_handle3
        global viewport_handle4

        camera_path = "/World/Camera"
        camera_prim = stage.DefinePrim(camera_path, "Camera")
        camera_path1 = "/World/Camera1"
        camera_prim1 = stage.DefinePrim(camera_path1, "Camera")
        camera_path2 = "/World/Camera2"
        camera_prim2 = stage.DefinePrim(camera_path2, "Camera")
        camera_path3 = "/World/Camera3"
        camera_prim3 = stage.DefinePrim(camera_path3, "Camera")
        camera_path4 = "/World/Camera4"
        camera_prim4 = stage.DefinePrim(camera_path4, "Camera")

        if index==0:
            viewport_handle = viewport_interface.create_instance()
            viewport_window = viewport_interface.get_viewport_window(viewport_handle)
            viewport_window.set_active_camera(camera_path)
            viewport_window.set_texture_resolution(1280,720)
            
            #viewport_window.set_window_pos(300, 500)
            viewport_window.set_camera_position(camera_path, 0, -300, 150.0, True)
            viewport_window.set_camera_target(camera_path, 0.0, 0.0, 0.0, True)
            #viewport_window.set_visible(False)

            viewport_handle1 = viewport_interface.create_instance()
            viewport_window1 = viewport_interface.get_viewport_window(viewport_handle1)
            viewport_window1.set_active_camera(camera_path1)
            viewport_window1.set_texture_resolution(1280,720)
            #viewport_window.set_window_pos(300, 500)
            viewport_window1.set_camera_position(camera_path1, 0, 300, 150.0, True)
            viewport_window1.set_camera_target(camera_path1, 0.0, 0.0, 0.0, True)
            
            

            viewport_handle2 = viewport_interface.create_instance()
            viewport_window2 = viewport_interface.get_viewport_window(viewport_handle2)
            viewport_window2.set_active_camera(camera_path2)
            viewport_window2.set_texture_resolution(1280,720)
            #viewport_window.set_window_pos(300, 500)
            viewport_window2.set_camera_position(camera_path2, 300, 0, 150.0, True)
            viewport_window2.set_camera_target(camera_path2, 0.0, 0.0, 0.0, True)
            #viewport_window2.set_visible(False)

            viewport_handle3 = viewport_interface.create_instance()
            viewport_window3 = viewport_interface.get_viewport_window(viewport_handle3)
            viewport_window3.set_active_camera(camera_path3)
            viewport_window3.set_texture_resolution(1280,720)
            #viewport_window.set_window_pos(300, 500)
            viewport_window3.set_camera_position(camera_path3, -300, 0, 150.0, True)
            viewport_window3.set_camera_target(camera_path3, 0.0, 0.0, 0.0, True)

            viewport_handle4 = viewport_interface.create_instance()
            viewport_window4 = viewport_interface.get_viewport_window(viewport_handle4)
            viewport_window4.set_active_camera(camera_path4)
            viewport_window4.set_texture_resolution(1280,720)
            #viewport_window.set_window_pos(300, 500)
            viewport_window4.set_camera_position(camera_path4, 0, 0, 400.0, True)
            viewport_window4.set_camera_target(camera_path4, 0.0, 0.0, 0.0, True)

        else:
            #viewport_handle = viewport_interface.get_instance('Viewport2')
            viewport_window = viewport_interface.get_viewport_window(viewport_handle)
            viewport_window.set_active_camera(camera_path)
            viewport_window.set_camera_position(camera_path, 0, -300, 150.0, True)
            viewport_window.set_camera_target(camera_path, 0.0, 0.0, 0.0, True)

            #viewport_handle1 = viewport_interface.get_instance('Viewport3')
            viewport_window1 = viewport_interface.get_viewport_window(viewport_handle1)
            viewport_window1.set_active_camera(camera_path1)
            viewport_window1.set_camera_position(camera_path1, 0, 300, 150.0, True)
            viewport_window1.set_camera_target(camera_path1, 0.0, 0.0, 0.0, True)

            #viewport_handle2 = viewport_interface.get_instance('Viewport4')
            viewport_window2 = viewport_interface.get_viewport_window(viewport_handle2)
            viewport_window2.set_active_camera(camera_path2)
            viewport_window2.set_camera_position(camera_path2, 300, 0, 150.0, True)
            viewport_window2.set_camera_target(camera_path2, 0.0, 0.0, 0.0, True)

            viewport_window3 = viewport_interface.get_viewport_window(viewport_handle3)
            viewport_window3.set_active_camera(camera_path3)
            viewport_window3.set_camera_position(camera_path3, -300, 0, 150.0, True)
            viewport_window3.set_camera_target(camera_path3, 0.0, 0.0, 0.0, True)

            viewport_window4 = viewport_interface.get_viewport_window(viewport_handle4)
            viewport_window4.set_active_camera(camera_path4)
            viewport_window4.set_camera_position(camera_path4, 0, 0, 400.0, True)
            viewport_window4.set_camera_target(camera_path4, 0.0, 0.0, 0.0, True)

        simulation_app.update()
       
      

        the_world.play()
        for i in range(150):
            the_world.step(render=True)

        sd_helper = SyntheticDataHelper()

      
        sensor_names = [
        "rgb",
        "depth",
        "semanticSegmentation",
        "depthLinear",
        "instanceSegmentation"
        ]
        sd_helper.initialize(sensor_names,viewport_window)
        sd_helper.initialize(sensor_names,viewport_window1)
        sd_helper.initialize(sensor_names,viewport_window2)
        sd_helper.initialize(sensor_names,viewport_window3)
        sd_helper.initialize(sensor_names,viewport_window4)

        camera=sd_helper.get_camera_params(viewport_window)
        camera1=sd_helper.get_camera_params(viewport_window1)
        camera2=sd_helper.get_camera_params(viewport_window2)
        camera3=sd_helper.get_camera_params(viewport_window3)
        camera4=sd_helper.get_camera_params(viewport_window4)

        pose = camera['pose']
        pose1 = camera1['pose']
        pose2 = camera2['pose']
        pose3 = camera3['pose']
        pose4 = camera4['pose']

        camera_pose=pose.T
        camera_pose1=pose1.T
        camera_pose2=pose2.T
        camera_pose3=pose3.T
        camera_pose4=pose4.T

        gt = sd_helper.get_groundtruth(sensor_names, viewport_window)
        gt1 = sd_helper.get_groundtruth(sensor_names, viewport_window1)
        gt2 = sd_helper.get_groundtruth(sensor_names, viewport_window2)
        gt3 = sd_helper.get_groundtruth(sensor_names, viewport_window3)
        gt4 = sd_helper.get_groundtruth(sensor_names, viewport_window4)

        name=self.data_root+'/'+stage_folder+"/"+stage_folder+".png"
        folder_name=self.data_root+'/'+stage_folder+"/"


        depth_linear = gt["depthLinear"]
        semantic_seg = gt["semanticSegmentation"]
        instance_seg = gt["instanceSegmentation"][0]
        intrinsic_matrix = get_intrinsic_matrix(viewport_window,sd_helper)
        extrinsic_matrix = get_extrinsic_matrix(viewport_window,sd_helper, meters=True)
        print(extrinsic_matrix)
        semantic_seg_ids = np.unique(semantic_seg)
        instance_seg_ids = np.unique(instance_seg)
        info=gt["instanceSegmentation"][1]


        depth_linear1 = gt1["depthLinear"]
        semantic_seg1 = gt1["semanticSegmentation"]
        instance_seg1 = gt1["instanceSegmentation"][0]
        intrinsic_matrix1 = get_intrinsic_matrix(viewport_window1,sd_helper)
        extrinsic_matrix1 = get_extrinsic_matrix(viewport_window1,sd_helper, meters=True)
        print(extrinsic_matrix1)
        semantic_seg_ids1 = np.unique(semantic_seg1)
        instance_seg_ids1 = np.unique(instance_seg1)
        info1=gt1["instanceSegmentation"][1]

        depth_linear2 = gt2["depthLinear"]
        semantic_seg2 = gt2["semanticSegmentation"]
        instance_seg2 = gt2["instanceSegmentation"][0]
        intrinsic_matrix2 = get_intrinsic_matrix(viewport_window2,sd_helper)
        extrinsic_matrix2 = get_extrinsic_matrix(viewport_window2,sd_helper, meters=True)
        print(extrinsic_matrix2)
        semantic_seg_ids2 = np.unique(semantic_seg2)
        instance_seg_ids2 = np.unique(instance_seg2)
        info2=gt2["instanceSegmentation"][1]

        depth_linear3 = gt3["depthLinear"]
        semantic_seg3 = gt3["semanticSegmentation"]
        instance_seg3 = gt3["instanceSegmentation"][0]
        intrinsic_matrix3 = get_intrinsic_matrix(viewport_window3,sd_helper)
        extrinsic_matrix3 = get_extrinsic_matrix(viewport_window3,sd_helper, meters=True)
        print(extrinsic_matrix3)
        semantic_seg_ids3 = np.unique(semantic_seg3)
        instance_seg_ids3 = np.unique(instance_seg3)
        info3=gt3["instanceSegmentation"][1]

        depth_linear4 = gt4["depthLinear"]
        semantic_seg4 = gt4["semanticSegmentation"]
        instance_seg4 = gt4["instanceSegmentation"][0]
        intrinsic_matrix4 = get_intrinsic_matrix(viewport_window4,sd_helper)
        extrinsic_matrix4 = get_extrinsic_matrix(viewport_window4,sd_helper, meters=True)
        print(extrinsic_matrix4)
        semantic_seg_ids4 = np.unique(semantic_seg4)
        instance_seg_ids4 = np.unique(instance_seg4)
        info4=gt4["instanceSegmentation"][1]

        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_points()

        camera_location=np.array([0, -300, 150.0])
        camera_location1=np.array([0, 300, 150.0])
        camera_location2=np.array([300, 0, 150.0])
        camera_location3=np.array([-300, 0, 150.0])
        camera_location4=np.array([0, 0, 400.0])

        point_cloud_dic=self.combine(stage_folder,folder_name,info,the_world,draw,instance_seg_ids,instance_seg,depth_linear,intrinsic_matrix,extrinsic_matrix,camera_location,info1,instance_seg_ids1,instance_seg1,depth_linear1,intrinsic_matrix1,extrinsic_matrix1,camera_location1,info2,instance_seg_ids2,instance_seg2,depth_linear2,intrinsic_matrix2,extrinsic_matrix2,camera_location2,info3,instance_seg_ids3,instance_seg3,depth_linear3,intrinsic_matrix3,extrinsic_matrix3,camera_location3,info4,instance_seg_ids4,instance_seg4,depth_linear4,intrinsic_matrix4,extrinsic_matrix4,camera_location4)

        #print(point_cloud_dic)
        #self.make_plots(gt, name_suffix=name)
        #viewport_window1.set_visible(False)
        #viewport_window.set_visible(False)
        #viewport_window2.set_visible(False)


        all_instance_seg_ids = np.unique(np.concatenate((instance_seg_ids, instance_seg_ids1,instance_seg_ids2,instance_seg_ids3,instance_seg_ids4)))

        #list_of_pcs = [point_cloud_dic[i] for i in all_instance_seg_ids if i != 0]

        #o3d.visualization.draw_geometries_with_custom_animation(list_of_pcs)    
        
        #print(all_instance_seg_ids)
        display=[]
        candidates={}
       
        hand_mesh=o3d.io.read_triangle_mesh("/home/juncheng/Documents/fetch_hand_axis.obj")

        for i in all_instance_seg_ids:
            #pcd,mesh_t,mesh_cylinder,mesh=[]
            mesh_final={}
            if i!=0:
                
                o3d.io.write_point_cloud(folder_name+info[i-1][3]+"_"+str(info[i-1][0])+".pcd",point_cloud_dic[i])

                #pts=point_cloud_dic[i].points
                #pcd=point_cloud_dic[i].voxel_down_sample(voxel_size=0.01)
                pcd=point_cloud_dic[i]
               #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.5),fast_normal_computation=False)
                #pcd.orient_normals_towards_camera_location(camera_location=camera_location4)



                #downpcd=pcd.voxel_down_sample(voxel_size=0.1)
                fps = FPS(np.asarray(pcd.points),100)
                index=fps.fit() 
                #print(index)

                
                #o3d.visualization.draw_geometries([downpcd])
                #print(pcd)
                #pcd,ind = pcd.remove_radius_outlier(nb_points=10, radius=5)
                #o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
                kdtree = o3d.geometry.KDTreeFlann(pcd)
                

                #downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=500),fast_normal_computation=False)
                #pcd.paint_uniform_color([0, 0, 1])


                #viewport = omni.kit.viewport.get_default_viewport_window()
                #camera=sd_helper.get_camera_params(viewport)
                #pose = camera['pose']
               # camera_pose=pose.T

                #print(camera_pose)
                #camera_matrix = np.linalg.inv(extrinsic_matrix)

                #downpcd.orient_normals_towards_camera_location(camera_location=np.array([150., 150., 150.]))
                
                #o3d.visualization.draw_geometries([pcd])

                points = np.asarray(pcd.points) # Nx3 np array of points
                #points1 = np.asarray(downpcd.points) 
                normals = np.asarray(pcd.normals) # Nx3 np array of normals\
                
             
                #index_list= np.arange(0, len(points))
                index_list=index
                #print(index_list)
                translation=[]
                rotation=[]
                translation_bad=[]
                rotation_bad=[]
                start=np.empty([1, 3])
                end=np.empty([1, 3])
                result=[]
                for index in index_list:
                    #if index % 50== 0:
                        query = points[index,:]
                        (num, indices, distances) = kdtree.search_radius_vector_3d(query,1)
                        #print(indices)
                        #print(index)
                       # print(index_list)
                        #item_list = index_list.tolist()
                        #list_to_remove = np.asarray(indices).tolist()

                        #final_list = list(set(item_list) - set(list_to_remove))
                        #unwanted = indices
                        #item_list = [e for e in index_list if e not in unwanted]
                        #print(item_list)
                        #to_be_removed_inds = indices
                        
                        #points = points[[x for x in range(len(points)) if x not in to_be_removed_inds]]
                        
                        #print(points)
                    
                        #print(len(normals))
                        if np.any(np.asarray(indices) >= len(normals)):
                            continue



                        N=np.einsum('ij,ik->jk', normals[indices,:], (normals[indices,:]))
                    
                        w,V=np.linalg.eigh(N)
                    
                        R=np.fliplr(V)
                    
                        V1=R[:,0]#normal
                
                        
                        check1=np.dot(V1, normals[index,:], out=None)
                        
                        if check1 <=0:
                            a=Rz(np.pi)
                            R= np.dot(R, a)
                           
                        
                        #np.asarray(pcd.colors)[indices[1:], :] = [1, 0, 0]


                        #mesh=o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1.0, cone_radius=1.5, cylinder_height=5.0)
                        #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
                        #kdtree2 = o3d.geometry.KDTreeFlann(downpcd)

                        (num1, indices1, distances1) = kdtree.search_radius_vector_3d(query,1.2)
                                                
                        T1 = np.eye(4)
                        T1[:3, :3] = R
                        T1[0, 3] = query[0]
                        T1[1, 3] = query[1]
                        T1[2, 3] = query[2]
                        
                        
                        #print(T1)
                        T2 = np.eye(4)
                        #T2[:3, :3] = R
                        T2[0, 3] = 2
                        T2[1, 3] = 0
                        T2[2, 3] = 0
                        
                        T3=np.matmul(T1, T2)
                        
                        #t=[query[0],query[1],query[2]]
                        t=[T3[0, 3],T3[1, 3],T3[2, 3]]
                        
                        
                        
                        
                        
                        normal_vector=R[:,0]
                        #the_world.play()
                        #print(check_raycast(t,normal_vector))
                        ind=0
                        #collision=False
                        #for value in (points[indices1,:]):
                        t1 = np.eye(4)
                        t1[:3, :3] = R
                        t1[0, 3] = query[0]
                        t1[1, 3] = query[1]
                        t1[2, 3] = query[2]
                        
                        T4=np.matmul(t1, T2)
                        
                        t=[query[0],query[1],query[2]]
                        t2=[T4[0, 3],T4[1, 3],T4[2, 3]]
                        

                      
                        collision_dis=[]
                        a1=Rz(np.pi)
                        R1= np.matmul(R, a1)
                        normal_vector1=R1[:,0]
                        
                        dis_c=[]
                        t=np.expand_dims(t, axis=0)
                        start=np.vstack((start,t))

                        
                        # physX query to detect number of hits for a cubic region
                        
                        hand_list="/home/juncheng/Documents/fetch_hand.usd"
                       # setup_task = asyncio.ensure_future(self.load_stage(hand_list))
                       # while not setup_task.done():
                            #simulation_app.update()

                        standoff_depth=[-5.5,0.5]
                        #standoff_depth=[0,0]
                        rayDir = carb.Float3(normal_vector[0], normal_vector[1], normal_vector[2])
                        GripperDir = carb.Float3(normal_vector1[0], normal_vector1[1], normal_vector1[2])
                        #print(mesh_normal)
                        mesh_normal = [rayDir[0], rayDir[1], rayDir[2]]
                        gripper_normal = [GripperDir[0], GripperDir[1], GripperDir[2]]
                        mesh_normal = np.expand_dims(mesh_normal, axis=0)
                        #t = np.expand_dims(t, axis=0)
                        #print(mesh_normal)
                        
                        #print(result['poses'])

                        if (check_raycast(t2,normal_vector))==None:
        
                            approach_sampler=SurfaceApproachSampler(standoff_depth, 0, 0.0, mesh_normal,gripper_normal,t)
                            result=approach_sampler.sample(2)
                            for grasp_num in range(len(result['poses'])):
                                #quaternion_wxyz=result['poses'][grasp_num][3:]
                                #quaternion_wxyz=[quaternion_xyzw[3],quaternion_xyzw[0],quaternion_xyzw[1],quaternion_xyzw[2]]
                                
                                #qz = tra.quaternion_about_axis(np.pi, [0,0,1])
                                #q = tra.quaternion_multiply(quaternion_xyzw, qz)
                                #quaternion_wxyz=q

                                #gripper_orientation=quat_to_rot_matrix(np.array(quaternion_wxyz))

                                #gripper_orientation2_rot= np.dot(gripper_orientation, tra.rotation_matrix(np.pi, [0,0,1])[:3,:3])

                                #quaternion_wxyz=euler_angles_to_quat(Rotation.random().as_euler('zyx', degrees=True))
                                #quaternion_wxyz=euler_angles_to_quat([90,86.42,-60],degrees=True)
                                #quaternion_xyzw=[quaternion_wxyz[1],quaternion_wxyz[2],quaternion_wxyz[3],quaternion_wxyz[0]]

                                #asset=create_prim(
                                   # prim_path="/World/hand", 
                                    #prim_type="hand",
                                   # scale = np.array([1,1,1]),
                                   # usd_path=hand_list,
                                    #position=[-31.16753,-1.05492,3.51197],
                                    #translation=[-31.16753,-1.05492,3.51197],
                                    #orientation=euler_angles_to_quat([90,-60,86.42],degrees=True)
                                   # orientation=[0.6895,0.20083,-0.67799,0.16507],
                                    #semantic_label=b
                               # )
                                #xform_prim = XFormPrim(asset.GetPath())

                                #xform_prim.set_world_pose(position =np.array([-31.16753,-1.05492,3.51197]),orientation=quaternion_wxyz)   
                                #p,o=xform_prim.get_world_pose()
                                #angle=quat_to_euler_angles(o,degrees=True)
                                #print(angle)

                                #stage = omni.usd.get_context().get_stage()
                                #result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
                                # Get the prim
                                #cube_prim = stage.GetPrimAtPath(path)
                                # Enable physics on prim
                                # If a tighter collision approximation is desired use convexDecomposition instead of convexHull
                                #utils.setRigidBody(cube_prim, "convexHull", False)
                                #stage = omni.usd.get_context().get_stage()
                                #mesh = UsdGeom.Mesh.Get(stage, "/World/hand")
                                #print(mesh.GetPath().pathString)
                                #simulation_app.update()   
                            
                                #overlap=overlap_check([1,1,1],[T4[0, 3],T4[1, 3],T4[2, 3]],[0,0,1,0])
                                #hand_path=PhysicsSchemaTools.encodeSdfPath(Sdf.Path("/World/hand/hand"))
                                #object_path=PhysicsSchemaTools.encodeSdfPath(Sdf.Path(f"/World/objects/object_{i-1}"))
                                #stage = omni.usd.get_context().get_stage()
                                #usdGeom1 = UsdGeom.Mesh.Get(stage, "/World/hand")
                                #hand_path=usdGeom1.GetPath().pathString
                                
                                #usdGeom2 = UsdGeom.Mesh.Get(stage, f"/World/objects/object_{i-1}")
                                #object_path=usdGeom2.GetPath().pathString

                                #print(hand_path[0],hand_path[1])
                                
                                
                                #check=overlap_check_mesh(hand_path[0],hand_path[1])

                                quaternion_wxyz=result['poses'][grasp_num][3:]
                                Rq=tra.quaternion_matrix(quaternion_wxyz)[:3, :3]
                                #ox_base_origin_translation=[5.5+12/2,0,0]
                                
                                box_base_T = np.eye(4)
                                #T2[:3, :3] = R
                                box_base_T[0:3, 3] = [5.5+(12/2),0,0]
                            

                                origin_T = np.eye(4)
                                origin_T[:3, :3] = Rq
                                origin_T[0:3, 3] = result['poses'][grasp_num][0:3]
                                #origin[1, 3] = suction_prim[1]
                               #origin[2, 3] = suction_prim[2]
                                
                                box_base_T=np.matmul(origin_T, box_base_T)
                                        

                                finger_left_T = np.eye(4)
                                #T2[:3, :3] = R
                                finger_left_T[0:3, 3] = [5.5/2,0,(-1.4/2)-(10/2)]
                                finger_left_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
                                finger_left_extent=[12/2,7.4/2,13/2]

                                finger_right_T = np.eye(4)
                                #T2[:3, :3] = R
                                finger_right_T[0:3, 3] = [5.5/2,0,(1.4/2)+(10/2)]
                                finger_right_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
                                finger_right_extent=[15.4/2,2.5/2,1.4/2]

                                finger_left_T=np.matmul(origin_T, finger_left_T)

                                finger_right_T=np.matmul(origin_T, finger_right_T)

                                
                                box_base_origin=box_base_T[0:3, 3]
                                box_base_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
                                box_base_extent=[12/2,7.4/2,13/2]

                                overlap_api1=overlap_check(box_base_extent,box_base_origin,box_base_rotation)
                                overlap1=overlap_api.collision_check()

                                overlap_api2=overlap_check(finger_right_extent,finger_right_T[0:3, 3],finger_right_rotation)
                                overlap2=overlap_api2.collision_check()

                                overlap_api3=overlap_check(finger_left_extent,finger_left_T[0:3, 3],finger_left_rotation)
                                overlap3=overlap_api3.collision_check()


                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                #print(overlap)

#                                tt=[T3[0, 3],T3[1, 3],T3[2, 3]]
                                #box_base_origin_translation=[5.5+12/2,0,0]



















                                
                                #name,hit=check.collision_check()
                                
                                
                                
                                #verlap=get_physx_scene_query_interface().overlap_mesh_any(hand_path[0],hand_path[1])
                                #print(R)
                                
                                #gripper_orientation=quat_to_rot_matrix(np.array(quaternion_wxyz))
                                
                                #a3=Rz(np.pi)
                                #gripper_orientation2= np.dot(gripper_orientation, tra.rotation_matrix(np.pi, [0,0,1])[:3,:3])


                                #print(gripper_orientation)
                        
                                #display.append(mesh)
                                #display.append(pcd)
                                #gripper_orientation = quaternion_matrix(result['poses'][grasp_num][3:])


                                #hitss=overlap.collision_check()
                                #print(name)
                                #time.sleep(10)
                                #print(name,hit)
                                if overlap1==False and overlap2==False  and overlap3==False:
                                    #mesh=plot_gripper_pro_max(result['poses'][grasp_num][0:3], gripper_orientation2, 8, 2, score=1, color=None)
                                    print("no_collision")
                                    T = np.eye(4)
                                    T[:3, :3] = Rq
                                    T[0, 3] = result['poses'][grasp_num][0]
                                    T[1, 3] = result['poses'][grasp_num][1]
                                    T[2, 3] = result['poses'][grasp_num][2]
                                    mesh_t = copy.deepcopy(hand_mesh).transform(T)
                                    line_t=o3d.geometry.LineSet.create_from_triangle_mesh(mesh_t)
                                    #display.append(mesh_t)
                                    #mesh=create_mesh_cylinder(gripper_orientation, [result['poses'][grasp_num][0],result['poses'][grasp_num][1],result['poses'][grasp_num][2]])
                                    #display.append(pcd)
                                    display.append(line_t)
                                    #display.append(line_t)

                                #for i in range(10):
                                        #simulation_app.update() 
                                #display.append(pcd)
                                #stage = omni.usd.get_context().get_stage()
                                #stage.RemovePrim("/World/hand")
                                simulation_app.update()   

                        """
                        HITS=all_hits([T4[0, 3],T4[1, 3],T4[2, 3]],normal_vector1)
                        if HITS.check_raycast2()== None:
                           continue
                        else:
                        #start=np.expand_dims(start, axis=0)
                        

                        #t=np.expand_dims(t, axis=0)
                       # start=np.vstack((start,t))


                        #start = o3d.geometry.PointCloud()
                        #print(t)
                        #start.points = o3d.utility.Vector3dVector(np.expand_dims(t, axis=0))
                        #start.paint_uniform_color([1, 0, 0])

                            k,dis1,position=HITS.check_raycast2()
                            print(k,position,dis1)
                            #if k!=f"/World/objects/object_{i-1}" or k=="/World/groundPlane/collisionPlane" or k=="/World/groundPlane/geom":
                            for instance in range(len(k)):
                                if k[instance]==f"/World/objects/object_{i-1}":
                                    if dis1[instance]>0.7:
                                        result.append(position[instance])


                            #position=np.expand_dims(position, axis=0)
                                        end=np.vstack((end,result))

                        """
                        #if k!=f"/World/objects/object_{i-1}" or k=="/World/groundPlane/collisionPlane" or k=="/World/groundPlane/geom":
                           # break2
                        #begin=[T4[0, 3],T4[1, 3],T4[2, 3]]
                        #position=[position[0],position[1],position[2]]
                       # begin==np.expand_dims(begin, axis=0)
                        #position=np.expand_dims(position, axis=0)

                        #position=np.vstack((begin,position))

                        #print(position)
                        #draw = _debug_draw.acquire_debug_draw_interface()
                        
                        #o3d.visualization.draw_geometries_with_custom_animation(display)
                        
                        #o3d.visualization.draw_geometries([pcd])
                       
                        
                        #end = o3d.geometry.PointCloud()
                        #end.points = o3d.utility.Vector3dVector(position)
                        #end.paint_uniform_color([0, 0, 1])
                        #start = o3d.geometry.PointCloud()
                        #start.points = o3d.utility.Vector3dVector(np.expand_dims([query[0],query[1],query[2]], axis=0))
                        #start.paint_uniform_color([1, 0, 0])
                        #display.append(start)
                        #display.append(pcd)
                        
                        #display.append(end)
                        #o3d.visualization.draw_geometries_with_custom_animation(display)
                        
                        #mesh=create_mesh_cylinder_detection(R, t,collision)
                        #draw_vertices_pcd = o3d.geometry.PointCloud()
                        #draw_vertices_pcd.points = o3d.utility.Vector3dVector(np.array(vertices))
                        


                        #mesh=create_mesh_cylinder(R, t)
                        #mesh_final[index]=mesh
                        #display.append(mesh)
                        #display.append(pcd)
                        #display.append(draw_vertices_pcd)
                        #display.append(draw_vertices_pcd_new)
                        #o3d.visualization.draw_geometries_with_custom_animation(display)

                        #if collision== False:

                            #display.append(draw_vertices_pcd)
                            #o3d.visualization.draw_geometries_with_custom_animation(display)

                            #rotation.append(R)
                            #translation.append(t)
                        #if collision== True:
                          #translation_bad.append(t)
                          #rotation_bad.append(R)
                        #o3d.visualization.draw_geometries_with_custom_animation([pcd])
                #print(start)
                start1 = o3d.geometry.PointCloud()
                start1.points = o3d.utility.Vector3dVector(start)
                start1.paint_uniform_color([0, 0, 1])

                end1 = o3d.geometry.PointCloud()
                end1.points = o3d.utility.Vector3dVector(end)
                end1.paint_uniform_color([1, 0, 0])

                #display.append(start1)
                display.append(pcd)
                
                #o3d.visualization.draw_geometries_with_custom_animation(display)
                #values = [mesh_final[i] for i in mesh_final.keys()]
                #print(rotation)
                #print(translation)
                #display.append(values)--
                #candidates[i]=dict()
                #print(len(translation))
                #candidates[i]["rotation"]=rotation
                #print(len(translation))
                #candidates[i]["translation"]=translation
                #candidates[i]["object_name"]=info[i-1][3]
                #candidates[i]["rotation_bad"]=rotation_bad
                #candidates[i]["translation_bad"]=translation_bad

                #with open(folder_name+stage_folder+"_candidates"+".pkl", "wb") as f:
                    #pickle.dump(candidates, f)
                #f.close()

                #display.append(pcd)
        
        o3d.visualization.draw_geometries_with_custom_animation(display)
        #stage = omni.usd.get_context().get_stage()
        #prim=stage_info.keys()
        #for i in prim:
            #simulation_world.reset()
            #stage.RemovePrim(i)
            #print(self._screws[i])
            #self.scene.remove_object(prim_path=self._screws[i])
        #stage.RemovePrim(camera_path)
        #stage.RemovePrim(camera_path1)
        #stage.RemovePrim(camera_path2)
        #stage.RemovePrim(i)
        #stage.RemovePrim(i)
        simulation_app.update()
        #simulation_app.close()
        #simulation_app=SimulationApp({"headless": False})
        return None

    def __len__(self):
        return len(self.stage_root)

data_root="/home/juncheng/Downloads/dataset"
from torch.utils.data import DataLoader


stage_root = os.listdir(data_root)
len(stage_root)

my_dataset = Dataset(data_root=data_root)

for i in range(len(stage_root)):
    #print(i)
    my_dataset.__getitem__(i)
    #simulation_app.close()
    #simulation_app=SimulationApp({"headless": False})
#my_dataset[2]

#the_world = World(stage_units_in_meters = 0.01)
#while simulation_app.is_running():
    #the_world.step(render=True)
    #if the_world.is_playing():
        #if the_world.current_time_step_index == 0:
            #continue
#list(my_dataset)
#train_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)

#print(my_dataset[1])
#simulation_app.close()














