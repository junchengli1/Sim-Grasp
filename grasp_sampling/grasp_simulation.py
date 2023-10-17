from pathlib import Path
import argparse
import copy
from omni.isaac.kit import SimulationApp

# Add argparse arguments
base_dir = Path(__file__).parent
parser = argparse.ArgumentParser("Sim-Suction Simulation Environments")
parser.add_argument('--headless', type=bool, default=True, help='headless')#make it True when gerneate dataset 
parser.add_argument('--RobotArm_path', type=str, default=(base_dir.parent / "Props/ur10_invisible_suction.usd").as_posix(), help='robot arm path')
parser.add_argument('--SuctionCup_path', type=str, default=(base_dir.parent / "Props/short_gripper.usd").as_posix(), help='robot suction cup path')
parser.add_argument('--data_path', type=str, default=(base_dir.parent / "synthetic_data").as_posix(), help='data path')
parser.add_argument('--start_stage', type=int,default=0, help='start stage number')
parser.add_argument('--end_stage', type=int,default=500, help='end stage number')
parser.add_argument('--grid_space', type=int,default=300, help='grid space for cloning')
parser.add_argument('--save_pkl_flag', type=bool, default=True, help='Save simulation evaluation results as pkl file')

args = parser.parse_args()

# Launch omniverse app
simulation_app = SimulationApp({"headless": args.headless})


# Standard libraries
import torch

import pickle
import os
import numpy as np
import math as m
import sys
# Typing
from typing import Optional, List
import typing
from pxr import UsdGeom, Usd

# Omniverse and Carb
import omni
import carb
from omni.isaac.core.utils.prims import get_prim_path

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView, XFormPrim,XFormPrimView
from omni.isaac.core.utils.prims import define_prim, delete_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.cloner import GridCloner
from omni.physx.scripts import utils
from pxr import UsdGeom
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, matrix_to_euler_angles
from omni.isaac.core.utils.nucleus import get_assets_root_path
from simulation_utils import *  # This imports all functions from the utility file
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.physx import get_physx_interface
from pxr import Sdf


def initialize_object_view(stage_ind, object_number):
    """Initialize object view for the current stage and object number."""
    return RigidPrimView(prim_paths_expr=f"/World/envs/env_0/stage_{stage_ind}_instanceable/objects/object_{object_number-1}", name="object_view")

def apply_rotation(rotation_candidate):
    """Apply the rotation to the candidate."""
    a = Rz(np.pi)
    return np.matmul(rotation_candidate, a)

def construct_transformation_matrix(rotation_candidate, translation_candidate, env_position):
    """Construct the transformation matrix."""
    translation_candidate = np.array(translation_candidate)
    env_position = np.array(env_position)

    T = np.eye(4)
    T[:3, :3] = rotation_candidate
    T[:3, 3] = translation_candidate + env_position
    return T

def handle_action_for_env(i,controller, rotation, translation, env_position, robot, articulation_controller):
    """Handle action for a single environment."""
    R = apply_rotation(rotation)
    transformation_matrix = construct_transformation_matrix(rotation, translation, env_position)
    
    # Set frame for visualization
    add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/target_{i}")
    orientation = euler_angles_to_quat(matrix_to_euler_angles(transformation_matrix[:3, :3]))
    frame = XFormPrim(f"/target_{i}", scale=[4, 4, 4])
    frame.set_world_pose(transformation_matrix[:3, 3], orientation)
    
    # Compute the transformation for placing position
    offset_transformation = np.eye(4)
    offset_transformation[0, 3] = 5
    final_transformation = np.matmul(transformation_matrix, offset_transformation)

    actions = controller.forward(
        picking_position=transformation_matrix[:3, 3],
        end_effector_orientation=euler_angles_to_quat(matrix_to_euler_angles(R)),
        placing_position=np.array([final_transformation[0, 3], final_transformation[1, 3], final_transformation[2, 3] + 10]),
        current_joint_positions=robot.get_joint_positions(),
        end_effector_offset=np.array([0, 0, 0]),
    )
    articulation_controller.apply_action(actions)

    return actions

def main_simulation_loop(break_out_flag,world,stage_ind, object_number, num_envs, envs_positions,new_candidates,good_suctions_translation,good_suctions_rotation,good_suctions_indices):

    """The main simulation loop."""
   # object_view = initialize_object_view(stage_ind, object_number)
    #pose_ori = object_view.get_world_poses()[0][0][2]
    
    #print(pose_ori)
    path_obj = f"/World/envs/env_0/stage_{stage_ind}_instanceable/objects/object_{object_number-1}"
    pose_ori=get_physx_interface().get_rigidbody_transformation(path_obj)["position"][2]
   
    while True:
        if world.is_stopped():
            break

        if not world.is_playing():
            world.step(render=not args.headless)
            continue
        
        stage = omni.usd.get_context().get_stage()
        groundPlane_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        xform_prim = XFormPrim(get_prim_path(groundPlane_prim))
        step_ind=0
        for mm in range(25):
                        xform_prim.set_world_pose(np.array([0,0,step_ind]))
                        world.step(render=not args.headless)
                        step_ind-=0.8

        for i in range(num_envs):
            
            # Check if the actions have been completed
            prim_paths=f"/World/envs/env_{i}/stage_{stage_ind}_instanceable/objects/object_{object_number-1}"
            pose=get_physx_interface().get_rigidbody_transformation(prim_paths)["position"][2]

            print(pose,pose_ori)
            if pose - pose_ori >= -10:
                print("success")
                ori_ind=good_suctions_indices[i]
                new_candidates[object_number]["grasp_samples"][ori_ind]["simulation_quality"]=1
                #rotation.append(rotation_candidates_after_seal[i])
                #translation.append(translation_candidates_after_seal[i])
            else:
                print("fail")
                ori_ind=good_suctions_indices[i]
                new_candidates[object_number]["grasp_samples"][ori_ind]["simulation_quality"]=0

            print("done picking and placing")

            # Exit if it's the last environment
            if i == num_envs - 1:
                break_out_flag = True
                #mass = object_view_current.get_masses(clone=False)                
                #new_candidates[object_number] = dict()
                #new_candidates[object_number]["rotation_after_exp_success"] = rotation
                #new_candidates[object_number]["translation_after_exp_success"] = translation
                #new_candidates[object_number]["rotation_after_exp_fail"] = rotation_bad
                #new_candidates[object_number]["translation_after_exp_fail"] = translation_bad
                #new_candidates[object_number]["object_mass"] = mass[0]

                if args.save_pkl_flag ==True:
                    with open(args.data_path+f"/stage_{stage_ind}/"+f"stage_{stage_ind}_grasp_simulation_candidates"+".pkl", "wb") as f:
                        pickle.dump(new_candidates, f)
                    f.close()
                break

        if break_out_flag:
            break
        world.step(render=not args.headless)
    

class EnvironmentSetup:
    def __init__(self, world):
        
        self.world = world

    def setup_environment(self):
        set_camera_view([250, 250, 550], [0.0, 0.0, 0.0])
        self.world.scene.clear()
        delete_prim("/World/envs")
        self.world.scene.add_default_ground_plane()


    def setup_prims(self, stage):
        curr_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        for prim in Usd.PrimRange(curr_prim):
            if (prim.IsA(UsdGeom.Xform)):
                if hasSchema(prim, "PhysicsRigidBodyAPI"):
                    pass
                else:
                    utils.setRigidBody(prim, "convexDecomposition", True)
        xform_prim = XFormPrim("/World/defaultGroundPlane")
            
        # Adjust the properties of the default ground plane
        xform_prim.set_local_scale([100,100,1])



class RobotSpawner:
    def __init__(self, world, robot_usd):
        self.world = world
        self.robot_usd = robot_usd
        self.robots=[]
        self.controllers=[]
        self.articulation_controllers=[]

    def spawn(self, num_envs,  good_suctions_translation,good_suctions_rotation, envs_positions):
        for env_i in range(num_envs):
            new_position = good_suctions_translation[env_i]
            new_position = np.add(new_position, np.array(envs_positions[env_i]))
            asset_path = "/media/juncheng/Disk4T/Sim-Grasp/Props/fetch.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path=f"/World/Robot_{env_i}")
            XFormPrim(
                            prim_path=f"/World/Robot_{env_i}",
                            name=f"ur10_{env_i}",
                            translation=new_position,
                            orientation=euler_angles_to_quat(np.array(matrix_to_euler_angles(good_suctions_rotation[env_i])))
                        )
                        
            #self.controllers.append(PickPlaceController(name="pick_place_controller", gripper = self.robots[env_i].gripper, robot_articulation = self.robots[env_i]))
                        
            #self.articulation_controllers.append(self.robots[env_i].get_articulation_controller())
        self.robots = XFormPrimView(prim_paths_expr="/World/Robot_.*", name="robot_view")
        self.world.scene.add(self.robots)
        return self.robots


def sim_suction_simulation():
    """
    Spawns the UR10 robot and clones cluttered environments using Isaac Sim Cloner API.
     
    The main function that drives the simulation of robot operations.
    
    For each stage (defined by start_stage and end_stage), the function:
    1. Loads candidate positions and orientations for the robot.
    2. Sets up the simulation environment.
    3. Clones the environments.
    4. Simulates robot actions in each environment.
    5. Determines success or failure of the robot's actions.
    6. Stores the results in a dictionary.
    7. Closes the simulation application after completing all stages.
    """
    
    # Iterate over the range of stages
    

    for stage_ind in range(args.start_stage,args.end_stage):
        # Flag to indicate if the current loop should exit
        break_out_flag=False

        # Print the current stage index for logging purposes
        print(stage_ind)

        # Construct the path to the candidate file for the current stage
        candidate=args.data_path+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}_grasp_candidates_after_overlap.pkl"
        
        # Load candidates for the current stage
        with open(candidate, 'rb') as f:
            candidates= pickle.load(f)
        
        # Dictionary to store new candidate results
        new_candidates={}
        new_candidates=copy.deepcopy(candidates)
        
        # Iterate over each candidate key
            # Lights-1
 
        for object_number in candidates.keys():
            
            # Initialize lists to store results for the current candidate
            rotation=[]
            translation=[]
            translation_bad=[]
            rotation_bad=[]

            # Create a new simulation stage
            create_new_stage() 

            world = World(stage_units_in_meters=0.01, physics_dt=0.01,rendering_dt=0.01,physics_prim_path="/physicsScene",backend="numpy")
            #world.get_physics_context().enable_gpu_dynamics(True)
            
            world.get_physics_context().enable_flatcache(True)
            
            carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

            env_setup = EnvironmentSetup(world)
            env_setup.setup_environment()
            env_setup.setup_prims(omni.usd.get_context().get_stage())

            
            translation_candidates_after_seal=candidates[object_number]["grasp_samples"]
            good_suctions_translation = [sample["suction_translation"] 
                 for sample in candidates[object_number]["grasp_samples"] 
                 if sample["collision_quality"] == 1]
            good_suctions_rotation = [sample["suction_rotation_matrix"] 
                 for sample in candidates[object_number]["grasp_samples"] 
                 if sample["collision_quality"] == 1]
            
            good_suctions_indices = [index 
                         for index, sample in enumerate(candidates[object_number]["grasp_samples"]) 
                         if sample["collision_quality"] == 1]
            
            #rotation_candidates_after_seal=candidates[object_number]["rotation_after_seal_pass"]
            num_envs = len(good_suctions_translation)
            #print(candidates[object_number]["grasp_samples"])
            print(f"Number of environments: {num_envs}")
            # If no candidates are available for the current stage, skip to the next iteration
            if num_envs==0:
                continue
            
            # Initialize the GridCloner for environment replication
          

            cloner = GridCloner(spacing=args.grid_space)
            cloner.define_base_env("/World/envs")
            # Everything under the namespace "/World/envs/env_0" will be cloned
            define_prim("/World/envs/env_0")

            # Set the cluttered asset path and add a reference to the stage

            cluttered_usd_path=args.data_path+f"/stage_{stage_ind}"+f"/stage_{stage_ind}_instanceable.usd"
            add_reference_to_stage(usd_path=cluttered_usd_path, prim_path=f"/World/envs/env_0/stage_{stage_ind}_instanceable",prim_type="Xform")
    
            #robot_cfg = UR10_CFG
            #robot = SingleArmManipulator(cfg=robot_cfg)
            #robot.spawn("/World/envs/env_0/Robot", translation=(0.0, 0.0, 0.0))

            #asset_path = "/media/juncheng/Disk4T/Sim-Grasp/Props/fetch.usd"
            #add_reference_to_stage(usd_path=asset_path, prim_path="/World/envs/env_0/Robot")

    
            # Clone the scene
            cloner.define_base_env("/World/envs")

            envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
            envs_positions = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
            #physics_scene_path = sim.get_physics_context().prim_path

            # Spawn robots in each cloned environmentc
            robot_spawner = RobotSpawner(world, args.RobotArm_path)

            robots = robot_spawner.spawn(num_envs, good_suctions_translation,good_suctions_rotation, envs_positions)

            # Reset the world state for simulation
            #robot.initialize("/World/envs/env_.*/Robot")
            # Reset states
            world.reset()

            #robot.reset_buffers()
            print("[INFO]: Setup complete...")
      
            main_simulation_loop(break_out_flag,world,stage_ind, object_number, num_envs, envs_positions,new_candidates,good_suctions_translation,good_suctions_rotation,good_suctions_indices)


           
            
            


if __name__ == "__main__":
    # Run the main function
    sim_suction_simulation()
    # Close the simulator
    simulation_app.close()
