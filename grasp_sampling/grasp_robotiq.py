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
from scipy.spatial.transform import Rotation as R
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

class PickPlaceController_base(BaseController):
    """ 
        A simple pick and place state machine for tutorials

        Each phase runs for 1 second, which is the internal time of the state machine

        Dt of each phase/ event step is defined

        - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
        - Phase 1: Lower end_effector down to encircle the target cube
        - Phase 2: Wait for Robot's inertia to settle.
        - Phase 3: close grip.
        - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
        - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
        - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
        - Phase 7: loosen the grip.
        - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
        - Phase 9: Move end_effector towards the old xy position.

        Args:
            name (str): Name id of the controller
            cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
            gripper (Gripper): a gripper controller for open/ close actions.
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            #self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008, 0.08]
            self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 1]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt)  != 6:
                raise Exception("events dt length must be less than 10")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="close")
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) or not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) == 6:
                raise Exception("events dt length must be less than 10")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase.
        """
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase.
        """
        self._pause = False
        return

        
class PickPlaceController(PickPlaceController_base):
    """[summary]

        Args:
            name (str): [description]
            surface_gripper (SurfaceGripper): [description]
            robot_articulation(Articulation): [description]
            events_dt (Optional[List[float]], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str,
        gripper: SurfaceGripper,
        robot_articulation: Articulation,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        if events_dt is None:
            #events_dt = [0.01, 0.0035, 0.01, 1.0, 0.008, 0.005, 0.005, 1, 0.01, 0.08]
            events_dt = None
        PickPlaceController_base.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation, attach_gripper=True
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        return

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            picking_position (np.ndarray): [description]
            placing_position (np.ndarray): [description]
            current_joint_positions (np.ndarray): [description]
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))
        return super().forward(
            picking_position,
            placing_position,
            current_joint_positions,
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
        )




class UR10(Robot):
    """[summary]

        Args:
            prim_path (str): [description]
            name (str, optional): [description]. Defaults to "ur10_robot".
            usd_path (Optional[str], optional): [description]. Defaults to None.
            position (Optional[np.ndarray], optional): [description]. Defaults to None.
            orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
            attach_gripper (bool, optional): [description]. Defaults to False.
            gripper_usd (Optional[str], optional): [description]. Defaults to "default".

        Raises:
            NotImplementedError: [description]
        """

    def __init__(
        self,
        prim_path: str,
        name: str = "ur10_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: bool = False,
        gripper_usd: Optional[str] = "default",
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                usd_path = assets_root_path + "/Isaac/Robots/UR10/ur10.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        else:
            # TODO: change this
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        self._gripper_usd = gripper_usd
        if attach_gripper:
            if gripper_usd == "default":
                #print("sdssssssssssssss")
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                #gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
                gripper_usd='/home/juncheng/Documents/roq.usd'
                add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=16.12, direction="x"
                )
            elif gripper_usd is None:
                carb.log_warn("Not adding a gripper usd, the gripper already exists in the ur10 asset")
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=16.12, direction="x"
                )
            else:
                raise NotImplementedError
        self._attach_gripper = attach_gripper
        return

    @property
    def attach_gripper(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self._attach_gripper

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> SurfaceGripper:
        """[summary]

        Returns:
            SurfaceGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        if self._attach_gripper:
            self._gripper.initialize(physics_sim_view=physics_sim_view, articulation_num_dofs=self.num_dof)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self.disable_gravity()
        self._end_effector.initialize(physics_sim_view)
        return

    def post_reset(self) -> None:
        Robot.post_reset(self)
        self._end_effector.post_reset()
        self._gripper.post_reset()
        return


async def load_stage(path):
        await omni.usd.get_context().open_stage_async(path)

def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
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

for s in range(30,500):
    #omni.physx.get_physx_interface().overwrite_gpu_setting(1)
    #omni.physx.get_physx_interface().start_simulation()
    break_out_flag=False
    #data_root="/home/juncheng/Downloads/dataset"
    #stage_root = os.listdir(data_root)
    #candidate="/home/juncheng/Downloads/dataset/"+f"stage_{s}"+ f"/stage_{s}_candidates.pkl"
    candidate="/media/juncheng/jason-msral/iros/"+f"stage_{s}"+ f"/stage_{s}_candidates_after_seal.pkl"
    #pkl_list= "/home/juncheng/Downloads/revision/"+f"stage_{s}"+ f"/stage_{s}.pkl"
    usd_list= "/media/juncheng/jason-msral/iros/"+f"stage_{s}"+ f"/stage_{s}.usd"


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

        translation_candidates=candidates[j]["translation_after_seal_pass"]
        rotation_candidates=candidates[j]["rotation_after_seal_pass"]
        
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
        gripper_usd="/home/juncheng/Documents/Isaac/Robots/UR10/ur10_invisible_suction.usd"
        #gripper_usd="/home/juncheng/Documents/Isaac/Robots/UR10/ur10.usd"
        suction_usd="/home/juncheng/Documents/Isaac/Robots/UR10/simple_articulation.usd"
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
                    
                    aa=Rz(np.pi)
                    R= np.matmul(rotation_candidates[int(number-1)], aa)
                    original_t= translation_candidates[int(number-1)]
                    T1 = np.eye(4)
                    T1[:3, :3] =rotation_candidates[int(number-1)]
                    T1[0, 3] = original_t[0]
                    T1[1, 3] = original_t[1]
                    T1[2, 3] = original_t[2]

                    angle=(np.array(matrix_to_euler_angles(R)))

                    if abs(angle[1])<= 1.91986 and abs(angle[1])>= 1.1:
                        #print(T1)
                        T2 = np.eye(4)
                        #T2[:3, :3] = R
                        T2[0, 3] = 0
                        T2[1, 3] = 100
                        T2[2, 3] = 0

                        T3=np.matmul(T1, T2)
                        t=np.array([T3[0, 3],T3[1, 3],70])
                    else:
                        T2 = np.eye(4)
                        #T2[:3, :3] = R
                        T2[0, 3] = 100
                        T2[1, 3] = 0
                        T2[2, 3] = 0

                        T3=np.matmul(T1, T2)
                        
                        t=np.array([T3[0, 3],T3[1, 3],70])


                    robots.append(simulation_world.scene.add(UR10(
                        prim_path=f"/World/UR10_{int(number-1)}",
                        name="ur10"+ str(int(number-1)),
                        usd_path= gripper_usd,
                        position=np.add(np.array(t),np.array([200*xaxis,200*yaxis,0])), 
                        attach_gripper=True,
                        #end_effector_prim_name="ee_link",
                        #gripper_usd= suction_usd,
                        )))
                    
                        


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
                    aa=Rz(np.pi)
                    R= np.matmul(rotation_candidates[int(number-1)], aa)
                    original_t= translation_candidates[int(number-1)]
                    T1 = np.eye(4)
                    T1[:3, :3] =rotation_candidates[int(number-1)]
                    T1[0, 3] = original_t[0]
                    T1[1, 3] = original_t[1]
                    T1[2, 3] = original_t[2]

                    angle=(np.array(matrix_to_euler_angles(R)))

                    if abs(angle[1])<= 1.91986 and abs(angle[1])>= 1.1:
                        #print(T1)
                        T2 = np.eye(4)
                        #T2[:3, :3] = R
                        T2[0, 3] = 0
                        T2[1, 3] = 100
                        T2[2, 3] = 0

                        T3=np.matmul(T1, T2)
                        t=np.array([T3[0, 3],T3[1, 3],70])

                    else:
                        T2 = np.eye(4)
                        #T2[:3, :3] = R
                        T2[0, 3] = 100
                        T2[1, 3] = 0
                        T2[2, 3] = 0

                        T3=np.matmul(T1, T2)
                        
                        t=np.array([T3[0, 3],T3[1, 3],70])
                        

                    robots.append(simulation_world.scene.add(UR10(
                        prim_path=f"/World/UR10_{int(number-1)}",
                        name="ur10"+ str(int(number-1)),
                        usd_path= gripper_usd,
                        position=np.add(np.array(t),np.array([200*(xaxis+1),200*(reminder),0])), 
                        attach_gripper=True,
                        #end_effector_prim_name="ee_link",
                        #gripper_usd= suction_usd,
                        )))
                
                    
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
                    
                    aa=Rz(np.pi)
                    R= np.matmul(rotation_candidates[int(number-1)], aa)
                    original_t= translation_candidates[int(number-1)]
                    T1 = np.eye(4)
                    T1[:3, :3] =rotation_candidates[int(number-1)]
                    T1[0, 3] = original_t[0]
                    T1[1, 3] = original_t[1]
                    T1[2, 3] = original_t[2]

                    angle=(np.array(matrix_to_euler_angles(R)))

                    if abs(angle[1])<= 1.91986 and abs(angle[1])>= 1.1:
                        #print(T1)
                        T2 = np.eye(4)
                        #T2[:3, :3] = R
                        T2[0, 3] = 0
                        T2[1, 3] = 100
                        T2[2, 3] = 0

                        T3=np.matmul(T1, T2)
                        t=np.array([T3[0, 3],T3[1, 3],70])
                    else:
                        T2 = np.eye(4)
                        #T2[:3, :3] = R
                        T2[0, 3] = 100
                        T2[1, 3] = 0
                        T2[2, 3] = 0

                        T3=np.matmul(T1, T2)
                        
                        t=np.array([T3[0, 3],T3[1, 3],70])
                    robots.append(simulation_world.scene.add(UR10(
                        prim_path=f"/World/UR10_{int(number-1)}",
                        name="ur10"+ str(int(number-1)),
                        usd_path= gripper_usd,
                        position=np.add(np.array(t),np.array([200*xaxis,200*yaxis,0])), 
                        attach_gripper=True,
                        end_effector_prim_name="ee_link",
                        #gripper_usd= suction_usd,
                        )))            
                 

                    if number == N:
                        break

        stage.RemovePrim("/World/objects")

        simulation_world.reset()
        for i in (range(N)):
            print(robots[i])
            #robots[i].set_local_scale([100,100,100])
            robots[i].set_visibility(True)
            #controllers.append(PickPlaceController(name="pick_place_controller", gripper = robots[i].gripper, robot_articulation = robots[i]))
            controllers.append(PickPlaceController(name="pick_place_controller", gripper = robots[i].gripper, robot_articulation = robots[i]))

            
            articulation_controllers.append(robots[i].get_articulation_controller())



        #draw = _debug_draw.acquire_debug_draw_interface()
        #draw.clear_points()

        #for i in (range(N)):
            #draw.draw_points([(new_task[f"{i}"]["translation"][0],new_task[f"{i}"]["translation"][1],new_task[f"{i}"]["translation"][2])], [(0, 1, 0, 1)] , [8])


        actions=[]
        prim_b = stage.GetPrimAtPath(new_task[f"{i}"]["prim_name"])##id-1

        pose_ori = omni.usd.utils.get_world_transform_matrix(prim_b)

        while simulation_app.is_running():
            simulation_world.step(render=True)
            #simulation_world.render()
            if simulation_world.is_playing():
                if simulation_world.current_time_step_index == 0:
                    simulation_world.reset()
                    for i in (range(N)):
                        controllers[i].reset()
                for i in (range(N)):
                    a=Rz(np.pi)
                    R= np.matmul(rotation_candidates[i], a)

                    T1 = np.eye(4)
                    T1[:3, :3] =rotation_candidates[i]
                    T1[0, 3] = new_task[f"{i}"]["translation"][0]
                    T1[1, 3] = new_task[f"{i}"]["translation"][1]
                    T1[2, 3] = new_task[f"{i}"]["translation"][2]
                    
                    #print(new_task[f"{i}"]["prim_name"])
                    
                    #print(T1)
                    T2 = np.eye(4)
                    #T2[:3, :3] = R
                    T2[0, 3] = 0.1
                    T2[1, 3] = 0
                    T2[2, 3] = 0
                    
                    T3=np.matmul(T1, T2)
                    t=[T3[0, 3],T3[1, 3],T3[2, 3]]



                    T2[0, 3] = 5
                    T3=np.matmul(T1, T2)
                    t2=[T3[0, 3],T3[1, 3],T3[2, 3]]

                    #print(mass_api.GetMassAttr().Get())
                   
                                
                    actions=controllers[i].forward(
                        picking_position=np.array(t),
                        #picking_position2=np.array(t),
                        end_effector_orientation=euler_angles_to_quat(np.array(matrix_to_euler_angles(R))),
                        placing_position=np.array([T3[0, 3],T3[1, 3],T3[2, 3]+10]),
                        current_joint_positions=robots[i].get_joint_positions(),
                        end_effector_offset=np.array([0, 0, 0]),
                            )
                    articulation_controllers[i].apply_action(actions)
                
                
                    if controllers[i].is_done():
                                                
                        prim_b = stage.GetPrimAtPath(new_task[f"{i}"]["prim_name"])##id-1
                        pose = omni.usd.utils.get_world_transform_matrix(prim_b)
                        folder_root="/media/juncheng/jason-msral/iros/"+f"stage_{s}/"
                        #rigid_body = dc.get_rigid_body(new_task[f"{i}"]["prim_name"])
                        #object_pose=dc.get_rigid_body_pose(rigid_body).p
                        #print(object_pose,pose_ori[3,2])
                        if pose[3, 2]-pose_ori[3,2] >= 10:
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
                            #new_candidates[j]=dict()
                            rigid_body = dc.get_rigid_body(f"/World/task_0_0/object_{j-1}")
                            object_mass=dc.get_rigid_body_properties(rigid_body).mass
                            #print("mass", object_mass)
                            print(rotation,translation)
                            new_candidates[j]=dict()
                            new_candidates[j]["rotation_after_exp_success"]=rotation
                            new_candidates[j]["translation_after_exp_success"]=translation
                            new_candidates[j]["rotation_after_exp_fail"]=rotation_bad
                            new_candidates[j]["translation__after_exp_fail"]=translation_bad
                            new_candidates[j]["object_mass"]=object_mass
                            #print(new_candidates)
                            with open(folder_root+f"stage_{s}"+"_seal_simulation_candidates"+".pkl", "wb") as f:
                                pickle.dump(new_candidates, f)
                            f.close()

                            break
                        #if robots[i].gripper.is_closed():
                            #print("success")
                if break_out_flag:
                    break
                        #simulation_world.pause()
        #simulation_app.close()     