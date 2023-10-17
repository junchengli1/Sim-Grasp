from omni.isaac.kit import SimulationApp
#CONFIG = {"headless":False,"display_options":100000000,"renderer":"RayTracedLighting","anti_aliasing":0}
CONFIG = {"headless":False}

simulation_app=SimulationApp(launch_config=CONFIG)
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.tasks import BaseTask
from pxr import UsdPhysics
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.prims import create_prim

from omni.isaac.universal_robots.tasks import PickPlace
#from omni.isaac.universal_robots.controllers import PickPlaceController
import numpy as np
from omni.isaac.core import World   
from pxr import Gf, Sdf, UsdGeom, UsdShade, Semantics, UsdPhysics
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
import copy
import math as m
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.rotations import euler_angles_to_quat,matrix_to_euler_angles
from omni.isaac.universal_robots import UR10
#from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.prims import XFormPrim, RigidPrim

from omni.isaac.dynamic_control import _dynamic_control

import copy
import matplotlib.pyplot as plt
from einops import rearrange
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.syntheticdata import visualize
import open3d as o3d

from omni.isaac.debug_draw import _debug_draw

from typing import Optional
import asyncio
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.prims import get_prim_path
from typing import Optional
import numpy as np
from omni.isaac.core.robots.robot import Robot
#from omni.isaac.surface_gripper import SurfaceGripper
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import find_nucleus_server
import carb
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation
import omni.isaac.motion_generation as mg
#from omni.isaac.surface_gripper import SurfaceGripper
#from omni.isaac.universal_robots.controllers import GripperController
from omni.isaac.universal_robots.controllers import RMPFlowController
import numpy as np
from typing import Optional, List

from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
from omni.isaac.core.controllers import BaseGripperController
import typing

from typing import Optional
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
import carb
from omni.isaac.manipulators.grippers import ParallelGripper
from typing import Optional
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import carb

from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper_Properties
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import carb
import omni.kit.app

from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
#import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.universal_robots.controllers import RMPFlowController
import numpy as np
from typing import Optional, List


from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.physx.scripts import utils
from omni.isaac.franka import Franka
from omni.isaac.core.robots import Robot

from omni.isaac.franka.controllers import PickPlaceController
async def load_stage(path):
        await omni.usd.get_context().open_stage_async(path)

def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
def Ry(theta):
    return np.matrix ([[m.cos(theta),    0,      m.sin(theta)  ],
					[0,                     1,      0                   ],
					[-m.sin(theta),   0,      m.cos(theta)  ]
					])
def Rx(theta):
    return np.matrix ([[m.cos(theta),    -m.sin(theta),    0],
					[m.sin(theta),    m.cos(theta),     0],
					[0,                     0,                      1]])

from math import sqrt, floor
def isPerfect(N):
    if (sqrt(N) - floor(sqrt(N)) != 0):
        return False
    return True
 
# Function to find the closest perfect square
# taking minimum steps to reach from a number
def getClosestPerfectSquare(N):
    if (isPerfect(N)):
        #print(N, "0")
        return (N,"0","ROUND")
 
    # Variables to store first perfect
    # square number above and below N
    aboveN = -1
    belowN = -1
    n1 = 0
 
    # Finding first perfect square
    # number greater than N
    n1 = N + 1
    while (True):
        if (isPerfect(n1)):
            aboveN = n1
            break
        else:
            n1 += 1
 
    # Finding first perfect square
    # number less than N
    n1 = N - 1
    while (True):
        if (isPerfect(n1)):
            belowN = n1
            break
        else:
            n1 -= 1
             
    # Variables to store the differences
    diff1 = aboveN - N
    diff2 = N - belowN
 
    if (diff1 > diff2):
        return(belowN, diff2, 0)
    else:
        return(aboveN, diff1, 1)
def isPerfect(N):
    if (sqrt(N) - floor(sqrt(N)) != 0):
        return False
    return True
 
# Function to find the closest perfect square
# taking minimum steps to reach from a number
def getClosestPerfectSquare(N):
    if (isPerfect(N)):
        #print(N, "0")
        return (N,"0","ROUND")
 
    # Variables to store first perfect
    # square number above and below N
    aboveN = -1
    belowN = -1
    n1 = 0
 
    # Finding first perfect square
    # number greater than N
    n1 = N + 1
    while (True):
        if (isPerfect(n1)):
            aboveN = n1
            break
        else:
            n1 += 1
 
    # Finding first perfect square
    # number less than N
    n1 = N - 1
    while (True):
        if (isPerfect(n1)):
            belowN = n1
            break
        else:
            n1 -= 1
             
    # Variables to store the differences
    diff1 = aboveN - N
    diff2 = N - belowN
 
    if (diff1 > diff2):
        return(belowN, diff2, 0)
    else:
        return(aboveN, diff1, 1)
 
import sys


from pxr import UsdGeom, Usd

from math import sqrt, floor

data_root="/media/juncheng/jason-msral/iros"
stage_root = os.listdir(data_root)
#len(stage_root)

def hasSchema(prim, schemaName):
    schemas = prim.GetAppliedSchemas()
    for s in schemas:
        if s == schemaName:
            return True
    return False

for s in range(0,500):
    #omni.physx.get_physx_interface().overwrite_gpu_setting(1)
    #omni.physx.get_physx_interface().start_simulation()
    break_out_flag=False
    #data_root="/home/juncheng/Downloads/dataset"
    #stage_root = os.listdir(data_root)
    #candidate="/home/juncheng/Downloads/dataset/"+f"stage_{s}"+ f"/stage_{s}_candidates.pkl"
    candidate="/media/juncheng/jason-msral/iros/"+f"stage_{s}"+ f"/stage_{s}_grasp_candidates_after_overlap_roq.pkl"
    #pkl_list= "/home/juncheng/Downloads/revision/"+f"stage_{s}"+ f"/stage_{s}.pkl"
    usd_list= "/media/juncheng/jason-msral/iros/"+f"stage_{s}"+ f"/stage_{s}.usd"
    #hand_list="/home/juncheng/Documents/hand_new.usd"


    with open(candidate, 'rb') as f:
        candidates= pickle.load(f)

   
    new_candidates={}

    #new_candidates=copy.deepcopy(candidates)
    f.close()

    for j in candidates.keys():
        rotation=[]
        translation=[]
        translation_bad=[]
        rotation_bad=[]
        break_out_flag=False
        setup_task = asyncio.ensure_future(load_stage(usd_list))

        while not setup_task.done():
            simulation_app.update()


        simulation_world = World(stage_units_in_meters = 0.01)
        simulation_world.step(render=False)

        stage = omni.usd.get_context().get_stage()
        curr_prim = stage.GetPrimAtPath("/World/objects")

        for prim in Usd.PrimRange(curr_prim):
            if (prim.IsA(UsdGeom.Xform)):
                if hasSchema(prim, "PhysicsRigidBodyAPI"):
                   pass
                else:
                    utils.setRigidBody(prim, "convexDecomposition", True)
                #if prim.HasAPI(UsdPhysics.MassAPI):
                    #pass
                #else:
                mass_api=UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateDensityAttr(0.0001)
      

        for prim in Usd.PrimRange(curr_prim):
            if (prim.IsA(UsdGeom.Mesh)):
                #utils.setCollider(prim, approximationShape="convexDecomposition")
                utils.setCollider(prim, approximationShape="none")
        
        




        #for i in candidates.keys():

        translation_candidates=candidates[j]["translation_after_overlap_pass"]
        rotation_candidates=candidates[j]["rotation_after_overlap_pass"]
        
        #translation_candidates=candidates[j]["translation"]
        #rotation_candidates=candidates[j]["rotation"]
        
        dc = _dynamic_control.acquire_dynamic_control_interface()
                    
                    #art = dc.get_articulation(f"/World/task_0_0/object_{j-1}")
                    #dc.wake_up_articulation(art)
                    #get_articulation_root_body
        

        # Driver code
        N = len(translation_candidates)
        
        if N==0:
            continue

        a,b,c=getClosestPerfectSquare(N)

        edge=m.sqrt(a)


        ground_prim = stage.GetPrimAtPath("/World/groundPlane/geom")
        xform_prim = XFormPrim(get_prim_path(ground_prim))
        xform_prim.set_local_scale(np.array([3,3,1]))
        xform_prim.set_world_pose(np.array([2820,2820,0]))

        new_task = dict()

        #result, nucleus_server = find_nucleus_server()
        #gripper_usd="/home/juncheng/Documents/Isaac/Robots/UR10/ur10_invisible_suction.usd"
        gripper_usd="/home/juncheng/Documents/Isaac/Robots/Franka/franka.usd"
        #gripper_usd="/home/juncheng/Documents/Isaac/Robots/UR10/ur10.usd"
        #suction_usd="/home/juncheng/Documents/Isaac/Robots/UR10/simple_articulation.usd"
        #suction_usd="/home/juncheng/Documents/hand_new.usd"
        #hand_list="/home/juncheng/Documents/robotiq_overlap_1.usd"
        hand_list="/home/juncheng/Documents/roq.usd"
        controllers = []
        articulation_controllers = []
        robots = []





        if c==0:
            for xaxis in range(int(edge)):
                for yaxis in range(int(edge)):
                    number=xaxis*edge+yaxis+1
                    omni.usd.duplicate_prim(stage,get_prim_path(curr_prim),f"/World/task_{xaxis}_{yaxis}")
                    tile_prim = stage.DefinePrim(f"/World/task_{xaxis}_{yaxis}", "Xform")
                    curr_prim = stage.GetPrimAtPath(f"/World/task_{xaxis}_{yaxis}")
                    xform_prim = XFormPrim(get_prim_path(curr_prim))
                    #print(np.array([200*xaxis,200*yaxis,0]))
                    xform_prim.set_world_pose(position = np.array([200*xaxis,200*yaxis,0]))
                    new_tran=np.add(np.array(translation_candidates[int(number-1)]),np.array([200*xaxis,200*yaxis,0]))
                    new_task[f"{int(number-1)}"]={"translation": new_tran,"translation_old": translation_candidates[int(number-1)],"orientation": rotation_candidates[int(number-1)],"prim_name":f"/World/task_{xaxis}_{yaxis}/object_{j-1}"}
                    
                   # aa=Rz(np.pi)
                    #R= np.matmul(rotation_candidates[int(number-1)], aa)
                    original_R=rotation_candidates[int(number-1)]
                    original_t= translation_candidates[int(number-1)]

                    asset_path = hand_list
                    add_reference_to_stage(usd_path=asset_path, prim_path=f"/World/UR10_{int(number-1)}")
                    

                    robots.append(simulation_world.scene.add(XFormPrim(prim_path=f"/World/UR10_{int(number-1)}",
                                                                   name="ur10"+ str(int(number-1)),
                                                                   translation=np.add(np.array(original_t),np.array([200*xaxis,200*yaxis,0])),
                                                                   orientation=euler_angles_to_quat(np.array(matrix_to_euler_angles(original_R)))
                                                                   )
                
                                                                   )
                                                                   )

                    
                        


            for reminder in range(int(b)):
                    number=a+reminder+1
                    omni.usd.duplicate_prim(stage,get_prim_path(curr_prim),f"/World/task_{xaxis+1}_{reminder}")
                    tile_prim = stage.DefinePrim(f"/World/task_{xaxis+1}_{reminder}", "Xform")
                    curr_prim = stage.GetPrimAtPath(f"/World/task_{xaxis+1}_{reminder}")
                    xform_prim = XFormPrim(get_prim_path(curr_prim))
                    #print(np.array([200*xaxis,200*yaxis,0]))
                    xform_prim.set_world_pose(position = np.array([200*(xaxis+1),200*(reminder),0]))
                    new_tran=np.add(np.array(translation_candidates[int(number-1)]),np.array([200*(xaxis+1),200*(reminder),0]))
                    new_task[f"{int(number-1)}"]={"translation": new_tran,"translation_old": translation_candidates[int(number-1)],"orientation": rotation_candidates[int(number-1)],"prim_name":f"/World/task_{xaxis+1}_{reminder}/object_{j-1}"}
                   #R= np.matmul(rotation_candidates[int(number-1)], aa)
                    original_R=rotation_candidates[int(number-1)]
                    original_t= translation_candidates[int(number-1)]
                    asset_path = hand_list
                    add_reference_to_stage(usd_path=asset_path, prim_path=f"/World/UR10_{int(number-1)}")
                    robots.append(simulation_world.scene.add(XFormPrim(prim_path=f"/World/UR10_{int(number-1)}",
                                                                   name="ur10"+ str(int(number-1)),
                                                                   translation=np.add(np.array(original_t),np.array([200*(xaxis+1),200*(reminder),0])),
                                                                   orientation=euler_angles_to_quat(np.array(matrix_to_euler_angles(original_R)))
                                                                   )
                
                                                                   )
                                                                   )
                    
        else:
            for xaxis in range(int(edge)):
                for yaxis in range(int(edge)):
                    number=xaxis*edge+yaxis+1
                    omni.usd.duplicate_prim(stage,get_prim_path(curr_prim),f"/World/task_{xaxis}_{yaxis}")
                    tile_prim = stage.DefinePrim(f"/World/task_{xaxis}_{yaxis}", "Xform")
                    curr_prim = stage.GetPrimAtPath(f"/World/task_{xaxis}_{yaxis}")
                    xform_prim = XFormPrim(get_prim_path(curr_prim))
                    #print(np.array([200*xaxis,200*yaxis,0]))
                    xform_prim.set_world_pose(position = np.array([200*xaxis,200*yaxis,0]))
                    new_tran=np.add(np.array(translation_candidates[int(number-1)]),np.array([200*xaxis,200*yaxis,0]))
                    new_task[f"{int(number-1)}"]={"translation": new_tran,"translation_old": translation_candidates[int(number-1)],"orientation": rotation_candidates[int(number-1)],"prim_name":f"/World/task_{xaxis}_{yaxis}/object_{j-1}"}
                    
                    original_R=rotation_candidates[int(number-1)]


                    original_t= translation_candidates[int(number-1)]
                    asset_path = hand_list
                    add_reference_to_stage(usd_path=asset_path, prim_path=f"/World/UR10_{int(number-1)}")
                    robots.append(simulation_world.scene.add(XFormPrim(prim_path=f"/World/UR10_{int(number-1)}",
                                                                   name="ur10"+ str(int(number-1)),
                                                                   translation=np.add(np.array(original_t),np.array([200*xaxis,200*yaxis,0])),
                                                                   orientation=euler_angles_to_quat(np.array(matrix_to_euler_angles(original_R)))
                                                                   )
                                                                   )
                                                                   )
                    if number == N:
                        break

        stage.RemovePrim("/World/objects")

        simulation_world.reset()
        """
        for i in (range(N)):
            print(robots[i])
            #robots[i].set_local_scale([100,100,100])
            robots[i].set_visibility(True)
            #controllers.append(PickPlaceController(name="pick_place_controller", gripper = robots[i].gripper, robot_articulation = robots[i]))
            controllers.append(PickPlaceController(name="pick_place_controller", gripper = robots[i].gripper, robot_articulation = robots[i]))

            
            articulation_controllers.append(robots[i].get_articulation_controller())

        """
        dc = _dynamic_control.acquire_dynamic_control_interface()
# Get handle to articulation

        #art = dc.get_articulation("/World/UR10_2/hand/base")


        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_points()

        for i in (range(N)):
            draw.draw_points([(new_task[f"{i}"]["translation"][0],new_task[f"{i}"]["translation"][1],new_task[f"{i}"]["translation"][2])], [(0, 1, 0, 1)] , [8])


        actions=[]
        prim_b = stage.GetPrimAtPath(new_task[f"{i}"]["prim_name"])##id-1

        pose_ori = omni.usd.utils.get_world_transform_matrix(prim_b)


        #rigidbody.CreateVelocityAttr(0,0,10)

        while simulation_app.is_running():
            simulation_world.step(render=True)
            #simulation_world.render()
            if simulation_world.is_playing():
                if simulation_world.current_time_step_index == 0:
                    simulation_world.reset()
                    #for i in (range(N)):
                        #controllers[i].reset()
                
                groundPlane_prim = stage.GetPrimAtPath("/World/groundPlane")
                xform_prim = XFormPrim(get_prim_path(groundPlane_prim))
                step_ind=0

                for kk in range(100):
                        simulation_world.step(render=False)


                for mm in range(200):
                         
                         xform_prim.set_world_pose(np.array([0,0,step_ind]))
                         simulation_world.step(render=True)
                         step_ind-=0.1
                
                for i in (range(N)):
                    
                    #new_R=np.hstack((-R[:,1],R[:,2],-R[:,0]))
                    #a=Ry(np.pi/2)
                    #a1=Rz(np.pi/2)
                    #aaa=np.matmul(a1, a)
                    #R= np.matmul(rotation_candidates[i], a)

                    T1 = np.eye(4)
                    T1[:3, :3] =rotation_candidates[i]
                    T1[0, 3] = new_task[f"{i}"]["translation"][0]
                    T1[1, 3] = new_task[f"{i}"]["translation"][1]
                    T1[2, 3] = new_task[f"{i}"]["translation"][2]
                    
                    #print(new_task[f"{i}"]["prim_name"])
                    
                    #print(T1)
                    T2 = np.eye(4)
                    #T2[:3, :3] = R
                    T2[0, 3] =0
                    T2[1, 3] = 0
                    T2[2, 3] = 0
                    
                    T3=np.matmul(T1, T2)
                    t=[T3[0, 3],T3[1, 3],T3[2, 3]]



                    T2[0, 3] = 5
                    T3=np.matmul(T1, T2)
                    t2=[T3[0, 3],T3[1, 3],T3[2, 3]]

                    #print(mass_api.GetMassAttr().Get())
                    quat_xyzw=Rotation.from_matrix(rotation_candidates[i]).as_quat()
                    #curr_prim = stage.GetPrimAtPath("/World/objects")
                   
                   
                
                    prim_b = stage.GetPrimAtPath(new_task[f"{i}"]["prim_name"])
                    pose = omni.usd.utils.get_world_transform_matrix(prim_b)
                    folder_root="/media/juncheng/jason-msral/iros/"+f"stage_{s}/"
                    
                    #rigid_body = dc.get_rigid_body(new_task[f"{i}"]["prim_name"])
                    #pose=dc.get_rigid_body_pose(rigid_body).p
                    #print(object_pose,pose_ori[3,2])
                    #rigid_body = dc.get_rigid_body("/World/groundPlane")

                   # dc.set_rigid_body_disable_gravity(rigid_body,True)

                    #object_pose=dc.get_rigid_body_pose(rigid_body).p
                    #print(object_pose,pose_ori[3,2])
                    print(pose[3, 2]-pose_ori[3,2])
                    if pose[3, 2]-pose_ori[3,2] >= -10:
                        print("success")
                        rotation.append(new_task[f"{i}"]["orientation"])
                        translation.append((new_task[f"{i}"]["translation_old"]))
                    else:
                        print("fail")
                        rotation_bad.append(new_task[f"{i}"]["orientation"])
                        translation_bad.append((new_task[f"{i}"]["translation_old"]))

                    print("done picking and placing")
                    if i== N-1:
                            print(j)
                            break_out_flag = True
                         
                            new_candidates[j]=dict()
                            new_candidates[j]["rotation_after_exp_success"]=rotation
                            new_candidates[j]["translation_after_exp_success"]=translation
                            new_candidates[j]["rotation_after_exp_fail"]=rotation_bad
                            new_candidates[j]["translation__after_exp_fail"]=translation_bad
                           # new_candidates[j]["object_mass"]=object_mass
                            #print(new_candidates)
                            with open(folder_root+f"stage_{s}"+"_grasp_simulation_candidates_roq"+".pkl", "wb") as f:
                                pickle.dump(new_candidates, f)
                            f.close()

                            break











                  
                            #break
                        #if robots[i].gripper.is_closed():
                            #print("success")
                if break_out_flag:
                    break
                        #simulation_world.pause()
        #simulation_app.close()     