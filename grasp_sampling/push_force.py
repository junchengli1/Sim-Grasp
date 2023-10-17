from omni.isaac.kit import SimulationApp
simulation_app=SimulationApp({"headless": True})
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
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_rot_matrix
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
#    import compress_pickle
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.extensions import enable_extension
from omni.kit.viewport.utility import get_viewport_from_window_name
enable_extension("omni.kit.window.viewport")  # enable legacy viewport interface
from omni.kit.viewport.utility import get_active_viewport_window
from typing import Optional
#import blosc
import os
import torch
import torch.utils.data
import torchvision
import pickle
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.rotations import euler_angles_to_quat,matrix_to_euler_angles

import math as m
import omni.physx
from omni.physx import get_physx_scene_query_interface
from dgl.geometry import farthest_point_sampler

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
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage

import time
voxel_size = 0.02

max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

import math
import typing
import numpy as np
import cProfile

# omniverse
from pxr import Gf
from omni.syntheticdata import helpers
from pxr import Usd, UsdGeom, Gf
from omni.physx.scripts import utils
from semantics.schema.editor import PrimSemanticData
from omni.isaac.core.utils.semantics import get_semantics
from omni.isaac.occupancy_map import _occupancy_map
from omni.isaac.core.utils.nucleus import get_assets_root_path
from scipy.spatial.transform import Rotation
from omni.physx import get_physx_simulation_interface, get_physx_interface
from pxr import PhysicsSchemaTools, UsdUtils

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
from omni.isaac.core.robots import RobotView
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim,delete_prim
"""
Main
"""
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units,clear_stage,close_stage,create_new_stage

from omni.isaac.universal_robots.controllers import RMPFlowController
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
            #self._events_dt = [0.01, 0.01, 0.1, 0.1, 0.05, 0.1, 0.08]
            self._events_dt = [0.01, 0.01, 0.1, 0.1, 0.05, 0.1, 0.08]
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
            #target_joint_positions = self._gripper.forward(action="close")
            pass
        elif self._event == 1:
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
            #print(position_target)
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
        
        elif self._event == 4:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=placing_position, target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 5:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 6:
            target_joint_positions = self._gripper.forward(action="open")

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
            #print(position_target)
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
            return 0
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
class SurfaceGripper(Gripper):
    """Provides high level functions to set/ get properties and actions of a surface gripper 
        (a suction cup for example).

        Args:
            end_effector_prim_path (str): prim path of the Prim that corresponds to the gripper root/ end effector.
            translate (float, optional): _description_. Defaults to 0.
            direction (str, optional): _description_. Defaults to "x".
            grip_threshold (float, optional): _description_. Defaults to 0.01.
            force_limit (float, optional): _description_. Defaults to 1.0e6.
            torque_limit (float, optional): _description_. Defaults to 1.0e4.
            bend_angle (float, optional): _description_. Defaults to np.pi/24.
            kp (float, optional): _description_. Defaults to 1.0e2.
            kd (float, optional): _description_. Defaults to 1.0e2.
            disable_gravity (bool, optional): _description_. Defaults to True.
        """

    def __init__(
        self,
        end_effector_prim_path: str,
        translate: float = 0,
        direction: str = "x",
        grip_threshold: float = 2,  
        force_limit: float = 300000000,  ###### newton*100
        torque_limit: float = 10000000000,
        bend_angle: float = np.pi /2,   
        kp: float = 1.0e4,
        kd: float = 1.0e3, 
        disable_gravity: bool = False,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)
        self._dc_interface = _dynamic_control.acquire_dynamic_control_interface()
        self._translate = translate
        self._direction = direction
        self._grip_threshold = grip_threshold
        self._force_limit = force_limit
        self._torque_limit = torque_limit
        self._bend_angle = bend_angle
        self._kp = kp
        self._kd = kd
        self._disable_gravity = disable_gravity
        self._virtual_gripper = None
        self._articulation_num_dofs = None
        return

    def initialize(
        self, physics_sim_view: omni.physics.tensors.SimulationView = None, articulation_num_dofs: int = None
    ) -> None:
        """Create a physics simulation view if not passed and creates a rigid prim view using physX tensor api.
            This needs to be called after each hard reset (i.e stop + play on the timeline) before interacting with any
            of the functions of this class.

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None
            articulation_num_dofs (int, optional): num of dofs of the Articulation. Defaults to None.
        """
        Gripper.initialize(self, physics_sim_view=physics_sim_view)
        self._articulation_num_dofs = articulation_num_dofs
        virtual_gripper_props = Surface_Gripper_Properties()
        virtual_gripper_props.parentPath = self._end_effector_prim_path
        virtual_gripper_props.d6JointPath = virtual_gripper_props.parentPath + "/d6FixedJoint"
        virtual_gripper_props.gripThreshold = self._grip_threshold
        virtual_gripper_props.forceLimit = self._force_limit
        virtual_gripper_props.torqueLimit = self._torque_limit
        virtual_gripper_props.bendAngle = self._bend_angle
        virtual_gripper_props.stiffness = self._kp
        virtual_gripper_props.damping = self._kd
        virtual_gripper_props.disableGravity = self._disable_gravity
        tr = _dynamic_control.Transform()
        if self._direction == "x":
            tr.p.x = self._translate
        elif self._direction == "y":
            tr.p.y = self._translate
        elif self._direction == "z":
            tr.p.z = self._translate
        else:
            carb.log_error("Direction specified for the surface gripper doesn't exist")
        virtual_gripper_props.offset = tr
        virtual_gripper = Surface_Gripper(self._dc_interface)
        virtual_gripper.initialize(virtual_gripper_props)
        self._virtual_gripper = virtual_gripper
        if self._default_state is None:
            self._default_state = not self.is_closed()
        return

    def close(self) -> None:
        """Applies actions to the articulation that closes the gripper (ex: to hold an object).
        """
        if not self.is_closed():
            self._virtual_gripper.close()
        if not self.is_closed():
            carb.log_warn("gripper didn't close successfully")
        return

    def open(self) -> None:
        """Applies actions to the articulation that opens the gripper (ex: to release an object held).
        """
        result = self._virtual_gripper.open()
        if not result:
            carb.log_warn("gripper didn't open successfully")

        return

    def update(self) -> None:
        self._virtual_gripper.update()
        return

    def is_closed(self) -> bool:
        return self._virtual_gripper.is_closed()

    def set_translate(self, value: float) -> None:
        self._translate = value
        return

    def set_direction(self, value: float) -> None:
        self._direction = value
        return

    def set_force_limit(self, value: float) -> None:
        self._force_limit = value
        return

    def set_torque_limit(self, value: float) -> None:
        self._torque_limit = value
        return

    def set_default_state(self, opened: bool):
        """Sets the default state of the gripper

        Args:
            opened (bool): True if the surface gripper should start in an opened state. False otherwise.
        """
        self._default_state = opened
        return

    def get_default_state(self) -> dict:
        """Gets the default state of the gripper

        Returns:
            dict: key is "opened" and value would be true if the surface gripper should start in an opened state. False otherwise.
        """
        return {"opened": self._default_state}

    def post_reset(self):
        Gripper.post_reset(self)
        if self._default_state:  # means opened is true
            self.open()
        else:
            self.close()
        return

    def forward(self, action: str) -> ArticulationAction:
        """calculates the ArticulationAction for all of the articulation joints that corresponds to "open"
           or "close" actions.

        Args:
            action (str): "open" or "close" as an abstract action.

        Raises:
            Exception: _description_

        Returns:
            ArticulationAction: articulation action to be passed to the articulation itself
                                (includes all joints of the articulation).
        """
        if self._articulation_num_dofs is None:
            raise Exception(
                "Num of dofs of the articulation needs to be passed to initialize in order to use this method"
            )
        if action == "open":
            self.open()
        elif action == "close":
            self.close()
        else:
            raise Exception("action {} is not defined for SurfaceGripper".format(action))
        return ArticulationAction(joint_positions=[None] * self._articulation_num_dofs)
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
                usd_path = assets_root_path + "/Isaac/Robots/UR10/ur3.usd"
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
                gripper_usd='/home/ubuntu/Documents/Props/short_gripper.usd'
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
    return np.array(list(gf_matrix))
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
    return [x,y,z]

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
def range_with_floats(start, stop, step):
    while stop > start:
        yield start
        start += step
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
    #print(intrinsic_matrix)
    return intrinsic_matrix

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

def check_raycast(origin,rayDir,draw,flag):
    # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
    # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
    origin = carb.Float3(origin[0], origin[1], origin[2])
    rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
    distance = 1800.0
    # physX query to detect closest hit
    hit = get_physx_scene_query_interface().raycast_closest(origin, rayDir, distance)
    if(hit["hit"]):
        stage = omni.usd.get_context().get_stage()
        # Change object colour to yellow and record distance from origin
        usdGeom = UsdGeom.Mesh.Get(stage, hit["rigidBody"])
        hitColor = Vt.Vec3fArray([Gf.Vec3f(255.0 / 255.0, 255.0 / 255.0, 0.0)])
        #usdGeom.GetDisplayColorAttr().Set(hitColor)
        distance = hit["distance"]
        #hit_position=[hit.position[0],hit.position[1],hit.position[2]]
        hit_position=[hit["position"][0],hit["position"][1],hit["position"][2]]
        hit_position_draw = carb.Float3(hit_position[0], hit_position[1], hit_position[2])
        if flag==True:
            draw.draw_lines([origin], [hit_position_draw], [carb.Float4(0,0,1,0.1)], [15])
            draw.draw_points([hit_position_draw],[carb.Float4(1,0,0,1)] , [15])
            draw.draw_points([origin],[carb.Float4(0,0,1,1)] , [15])

        return usdGeom.GetPath().pathString, distance,hit_position
    return None, None,None

def check_raycast2(origin,rayDir):
    # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
    # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
    origin = carb.Float3(origin[0], origin[1], origin[2])
    rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
    distance = 80.0
    # physX query to detect closest hit
    hit = get_physx_scene_query_interface().raycast_closest(origin, rayDir, distance,True)
    if(hit["hit"]):
        stage = omni.usd.get_context().get_stage()
        # Change object colour to yellow and record distance from origin
        usdGeom = UsdGeom.Mesh.Get(stage, hit["rigidBody"])
        hitColor = Vt.Vec3fArray([Gf.Vec3f(255.0 / 255.0, 255.0 / 255.0, 0.0)])
        #usdGeom.GetDisplayColorAttr().Set(hitColor)
        distance = hit["distance"]
        hit_position=[hit["position"][0],hit["position"][1],hit["position"][2]]


        return usdGeom.GetPath().pathString,hit_position
    return None
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

def create_mesh_cylinder(R, t, radius=0.1, height=1):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    cylinder.vertices = o3d.utility.Vector3dVector(np.array(vertices))
   # print(cylinder.vertices)
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

def isaac_visual(vertices, ploygon,draw,dic_c_as_a_list,collision):
    for circle_i in range(vertices.shape[0]):

        for connect_ind in range(ploygon-1):
            if collision==True:
                color=[carb.Float4(1,0,0,1)]
            if collision==False:
                color=[carb.Float4(0,1,0,1)]

            begin=carb.Float3(np.array(vertices)[circle_i][connect_ind][0],np.array(vertices)[circle_i][connect_ind][1],np.array(vertices)[circle_i][connect_ind][2])
            end=carb.Float3(np.array(vertices)[circle_i][connect_ind+1][0],np.array(vertices)[circle_i][connect_ind+1][1],np.array(vertices)[circle_i][connect_ind+1][2])
        
            begin_after=dic_c_as_a_list[circle_i][connect_ind]
            end_after=dic_c_as_a_list[circle_i][connect_ind+1]
            
            draw.draw_lines([begin], [end], [carb.Float4(0,0,1,1)], [1])
            draw.draw_lines([begin_after], [end_after], color, [8])
            
            #print(connect_ind,connect_ind+1)

            if connect_ind==ploygon-2:

                begin_last=carb.Float3(np.array(vertices)[circle_i][0][0],np.array(vertices)[circle_i][0][1],np.array(vertices)[circle_i][0][2])
                end_last=carb.Float3(np.array(vertices)[circle_i][connect_ind+1][0],np.array(vertices)[circle_i][connect_ind+1][1],np.array(vertices)[circle_i][connect_ind+1][2])
                
                begin_after_last=dic_c_as_a_list[circle_i][0]
                end_after_last=dic_c_as_a_list[circle_i][connect_ind+1]
                
                draw.draw_lines([begin_last], [end_last], [carb.Float4(0,0,1,1)], [1])
                draw.draw_lines([begin_after_last], [end_after_last], color, [8])
            #print(connect_ind,len(dis_c))
            if (connect_ind+2)>= len(dic_c_as_a_list[circle_i]):
                break

                #print(connect_ind+1,"0")
        
        #ver_record_before=[]
        for connect_ind in range(len(dic_c_as_a_list[circle_i])):
                #print(len(dis_c[circle_i]))
                #print(connect_ind)
                if collision==True:
                    color=[carb.Float4(1,0,0,1)]
                if collision==False:
                    color=[carb.Float4(0,1,0,1)]

                suction_before=carb.Float3(np.array(vertices)[circle_i][connect_ind][0],np.array(vertices)[circle_i][connect_ind][1],np.array(vertices)[circle_i][connect_ind][2])
                suction_after=dic_c_as_a_list[circle_i][connect_ind]
                #ver_record_before.append(suction_before)
                #ver_record_after.append(suction_after)
                
                
                draw.draw_lines([suction_before], [suction_after], [carb.Float4(1,1,1,0.2)], [1])
    
        if circle_i>0:
            
            for connect_ind in range(len(dic_c_as_a_list[circle_i])):

                if collision==True:
                    color=[carb.Float4(1,0,0,1)]
                if collision==False:
                    color=[carb.Float4(0,1,0,1)]
                draw.draw_lines([carb.Float3(np.array(vertices)[circle_i][connect_ind][0],np.array(vertices)[circle_i][connect_ind][1],np.array(vertices)[circle_i][connect_ind][2])], [carb.Float3(np.array(vertices)[circle_i-1][connect_ind][0],np.array(vertices)[circle_i-1][connect_ind][1],np.array(vertices)[circle_i-1][connect_ind][2])], [carb.Float4(0,0,1,0.2)], [1])
                #print(circle_i,connect_ind)
                #print(dic_c_as_a_list[circle_i][connect_ind])
                if len(dic_c_as_a_list[circle_i])==len(dic_c_as_a_list[circle_i-1]):
                    draw.draw_lines([dic_c_as_a_list[circle_i][connect_ind]],[dic_c_as_a_list[circle_i-1][connect_ind]],color, [8])
                else:
                    continue
        
        
        #draw.clear_points()
def get_ray_directions(n_rays):
    # Generate random directions
    phi = (np.random.rand(n_rays) * 2 - 1) * np.pi
    costheta = np.random.rand(n_rays) * 2 - 1

    # Convert to spherical coordinates
    theta = np.arccos(costheta)

    # Convert to cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    directions = np.vstack([x, y, z])
    # Normalize the vectors to get direction vectors (unit vectors)
    directions = directions / np.linalg.norm(directions, axis=0)

    return directions.T
def add_cube(stage, path, size: float = 1, offset: Gf.Vec3d = Gf.Vec3d(0, 0, 0)):
    cubeGeom = UsdGeom.Cube.Define(stage, path)
    cubeGeom.CreateSizeAttr(size)
    cubeGeom.AddTranslateOp().Set(offset)

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
def get_gripper_pose(point, local_frame):
    """
    Compute the gripper pose at a specified point on the object surface.
    The z-axis of the gripper is aligned with the x-axis of the local frame.
    """
    # Position is directly the given point
    position = point

    # Compute the gripper orientation
    gripper_z_axis = local_frame[0, :]  # x-axis of local frame is z-axis of gripper
    gripper_x_axis = local_frame[1, :]  # y-axis of local frame is x-axis of gripper
    gripper_y_axis = local_frame[2, :]   # Ensure it's perpendicular to both x and z

    # Form the orientation matrix
    orientation = np.stack([gripper_x_axis, gripper_y_axis, gripper_z_axis], axis=-1)

    return position, orientation
def rotate_matrix(matrix, angle):
    """
    Rotate a matrix around its z-axis (x-axis in local frame) by the specified angle.
    """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])

    return np.dot(rotation_matrix, matrix)
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
            box_base_T[0:3, 3] = [0,0,22]
        
            
            box_base_T=np.matmul(origin_T, box_base_T)
                    

            finger_left_T = np.eye(4)
            #T2[:3, :3] = R
            finger_left_T[0:3, 3] = [-12.9,0,9]
            finger_left_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            finger_left_extent=[2,2,10/2]

            finger_right_T = np.eye(4)
            #T2[:3, :3] = R
            finger_right_T[0:3, 3] = [12.9,0,9]
            finger_right_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            finger_right_extent=[2,2,10/2]

            box_middle_T = np.eye(4)
            #T2[:3, :3] = R
            box_middle_T[0:3, 3] = [0,0,15.98]
            box_middle_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            box_middle_extent=[15,4.73,7]




            finger_left_T=np.matmul(origin_T, finger_left_T)

            finger_right_T=np.matmul(origin_T, finger_right_T)

            box_middle_T=np.matmul(origin_T, box_middle_T)

            
            box_base_origin=box_base_T[0:3, 3]
            box_base_rotation=Rotation.from_matrix(box_base_T[:3, :3]).as_quat()
            box_base_extent=[3,3,20]

            
            
            overlap_api1=overlap_check(box_base_extent,box_base_origin,box_base_rotation)
            overlap1=overlap_api1.collision_check()

            overlap_api2=overlap_check(finger_right_extent,finger_right_T[0:3, 3],finger_right_rotation)
            overlap2=overlap_api2.collision_check()

            overlap_api3=overlap_check(finger_left_extent,finger_left_T[0:3, 3],finger_left_rotation)
            overlap3=overlap_api3.collision_check()

            overlap_api4=overlap_check(box_middle_extent,box_middle_T[0:3, 3],box_middle_rotation)
            overlap4=overlap_api4.collision_check()


            #print(overlap1,overlap2,overlap3,overlap4)

            overlaps = [overlap1, overlap2, overlap3, overlap4]
    
            # Count the number of False occurrences
            false_count = overlaps.count(False)
            
            # Calculate the score
            collision_quality = false_count / len(overlaps)
            
           

            if overlap1==False and overlap2==False and overlap3==False  and overlap4==False:
                if visualize_flag ==True:
                    draw_rotated_box(draw,[0,0,22],box_base_extent,origin_T)
                    draw_rotated_box(draw,[-12.9,0,9],finger_left_extent,origin_T)
                    draw_rotated_box(draw,[12.9,0,9],finger_right_extent,origin_T)
                    draw_rotated_box(draw,[0,0,15.98],box_middle_extent,origin_T)

            if visualize_flag ==True:
                simulation_app.update()
                
                the_world.step(render=True)
                draw.clear_lines()


            return suction_translation, suction_rotation_matrix_new, collision_quality
            #add_reference_to_stage(get_assets_root_path(
            # ) + "/Isaac/Props/UIElements/frame_prim.usd", f"/target_{j}")

            #frame = XFormPrim(f"/target_{j}", scale=[4, 4, 4])
            # position = box_base_origin
            #orientation = euler_angles_to_quat(matrix_to_euler_angles(box_base_T[:3, :3]))
            # frame.set_world_pose(position, orientation)



            #print('position:', suction_translation)
            # print(orientation)

            #quaternion_wxyz=euler_angles_to_quat(Rotation.random().as_euler('zyx', degrees=True))

            #frame.set_world_pose(position, orientation)

def sample_push (rotation_angle,suction_translation,suction_rotation_matrix,draw,the_world,visualize_flag):
        #for i in range(num_samples):
        # Compute the rotation angle in radians
        #rotation_angle = 2 * np.pi * i / num_samples  # This will range from 0 to 2*pi

        # Construct the rotation matrix for this angle around the rotation axis
        # Define the rotation axis (which is the surface normal)
        rotation_axis = suction_rotation_matrix[:, 0]  # This is the first column of the matrix

        rotation_matrix = rotation_matrix_from_axis_and_angle(rotation_axis, rotation_angle)
        rotation_matrix = np.reshape(rotation_matrix, (3,3))

        #print('suction_rotation_matrix[:, 1] shape:', suction_rotation_matrix[:, 1].shape)

        
        suction_rotation_matrix_y = (suction_rotation_matrix[:, 1])

        # Apply the rotation to the push direction
        #print('rotation_matrix shape:', rotation_matrix.shape)


        #print('suction_rotation_matrix[:, 1] shape:', suction_rotation_matrix_y.shape)

        rotated_push_direction = np.matmul(rotation_matrix, suction_rotation_matrix_y)

        origin=suction_translation
        maxDist=5
        endPoint = carb.Float3(origin[0] + rotated_push_direction[0]*maxDist, origin[1] + rotated_push_direction[1]*maxDist, origin[2] + rotated_push_direction[2]*maxDist)

        #draw.draw_lines([suction_translation_draw_point], [endPoint], [carb.Float4(0,0,1,1)], [1])
        suction_translation_draw_point=carb.Float3(suction_translation[0],suction_translation[1],suction_translation[2])
        if visualize_flag==True:
            draw.draw_points([suction_translation_draw_point],[carb.Float4(1,0,0,1)] , [15])
            draw.draw_lines([suction_translation_draw_point], [endPoint], [carb.Float4(0,0,1,1)], [1])


        return origin, rotated_push_direction
        # Store the rotated push direction
        #push_direction_samples.append(rotated_push_direction)

    
    #draw.draw_lines([origin], [hit_position_draw], [carb.Float4(0,0,1,0.1)], [2])
        #draw.draw_points([suction_translation_draw_point],[carb.Float4(1,0,0,1)] , [15])

def createJoint(stage, joint_type, from_prim, to_prim):
    # for single selection use to_prim
    if to_prim is None:
        to_prim = from_prim
        from_prim = None

    from_path = from_prim.GetPath().pathString if from_prim is not None and from_prim.IsValid() else ""
    to_path = to_prim.GetPath().pathString if to_prim is not None and to_prim.IsValid() else ""
    single_selection = from_path == "" or to_path == ""

    # to_path can be not writable as in case of instancing, find first writable path
    joint_base_path = to_path
    base_prim = stage.GetPrimAtPath(joint_base_path)
    while base_prim != stage.GetPseudoRoot():
        if base_prim.IsInMaster():
            base_prim = base_prim.GetParent()
        elif base_prim.IsInstanceProxy():
            base_prim = base_prim.GetParent()
        elif base_prim.IsInstanceable():
            base_prim = base_prim.GetParent()
        else:
            break
    joint_base_path = str(base_prim.GetPrimPath())
    if joint_base_path == '/':
        joint_base_path = ''

    joint_name = "/" +("Fixed_2" + "Joint")
    joint_path = joint_base_path + joint_name

    if joint_type == "Fixed":
        component = UsdPhysics.FixedJoint.Define(stage, joint_path)
    elif joint_type == "Revolute":
        component = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        component.CreateAxisAttr("X")
    elif joint_type == "Prismatic":
        component = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        component.CreateAxisAttr("X")
    elif joint_type == "Spherical":
        component = UsdPhysics.SphericalJoint.Define(stage, joint_path)
        component.CreateAxisAttr("X")
    elif joint_type == "Distance":
        component = UsdPhysics.DistanceJoint.Define(stage, joint_path)
        component.CreateMinDistanceAttr(0.0)
        component.CreateMaxDistanceAttr(0.0)
    elif joint_type == "Gear":
        component = PhysxSchema.PhysxPhysicsGearJoint.Define(stage, joint_path)
    elif joint_type == "RackAndPinion":
        component = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(stage, joint_path)
    else:
        component = UsdPhysics.Joint.Define(stage, joint_path)
        prim = component.GetPrim()
        for limit_name in ["transX", "transY", "transZ", "rotX", "rotY", "rotZ"]:
            limit_api = UsdPhysics.LimitAPI.Apply(prim, limit_name)
            limit_api.CreateLowAttr(1.0)
            limit_api.CreateHighAttr(-1.0)

    xfCache = UsdGeom.XformCache()

    if not single_selection:
        to_pose = xfCache.GetLocalToWorldTransform(to_prim)
        from_pose = xfCache.GetLocalToWorldTransform(from_prim)
        rel_pose = to_pose * from_pose.GetInverse()
        rel_pose = rel_pose.RemoveScaleShear()
        pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
        rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())

        component.CreateBody0Rel().SetTargets([Sdf.Path(from_path)])
        component.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
        component.CreateLocalPos0Attr().Set(pos1)
        component.CreateLocalRot0Attr().Set(rot1)
        component.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    else:
        to_pose = xfCache.GetLocalToWorldTransform(to_prim)
        to_pose = to_pose.RemoveScaleShear()
        pos1 = Gf.Vec3f(to_pose.ExtractTranslation())
        rot1 = Gf.Quatf(to_pose.ExtractRotationQuat())

        component.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
        component.CreateLocalPos0Attr().Set(pos1)
        component.CreateLocalRot0Attr().Set(rot1)
        component.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    component.CreateBreakForceAttr().Set(1000000000000000)
    component.CreateBreakTorqueAttr().Set(100000000000000)

    return stage.GetPrimAtPath(joint_base_path + joint_name)            

class Dataset():
    def __init__(self, data_root):
        self.data_root = data_root
        self.pointcloud_root= "/media/juncheng/ubuntu_data1/home/ubuntu/Downloads/amazon/pointcloud"

        self.stage_root = os.listdir(data_root)


    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)
    

    def __getitem__(self, stage_ind):
        
        stage_folder=self.data_root+f"/stage_{stage_ind}"
        usd_list=self.data_root+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}.usd"
        #pkl_list=self.data_root+'/'+stage_folder+"/"+stage_folder+".pkl"

       # with open(pkl_list, 'rb') as f:
            #stage_info = pickle.load(f)
        #print(stage_info.keys())
        setup_task = asyncio.ensure_future(self.load_stage(usd_list))
        #simulation_world=World(stage_units_in_meters = 0.01)
        while not setup_task.done():
            simulation_app.update()
            #simulation_world.step()


        the_world = World(stage_units_in_meters = 0.01,backend="torch")
        #omni.isaac.core.utils.viewports.set_camera_view(eye=np.array([0, 0, 140]), target=np.array([0, 0, 0]))
        #set_camera_view(eye=np.array([0, 0, 140]), target=np.array([0, 0, 0]))
        stage = omni.usd.get_context().get_stage()

        with open("/media/juncheng/ubuntu_data1/home/ubuntu/Downloads/amazon/"+"seg_dic.pkl", "rb") as f:
                    #pickle.dump(seg_map, f)
                seg_dic=pickle.load(f)
        f.close()

        W, H = (1280, 720)
        

        simulation_app.update()
       
        dc = _dynamic_control.acquire_dynamic_control_interface()


        the_world.play()
        for i in range(150):
            the_world.step(render=True)

      
        curr_prim = stage.GetPrimAtPath("/World/objects")
        
        
        
        #pc_data = pcd_annotators.get_data()
        #print(pc_data)
        
        for prim in Usd.PrimRange(curr_prim):
            if (prim.IsA(UsdGeom.Xform)):
                #utils.setCollider(prim, approximationShape="convexDecomposition")
                utils.removeRigidBody(prim)
                #print(prim)
        
        #for i in range(150):
            #the_world.step(render=True)

        for prim in Usd.PrimRange(curr_prim):
            if (prim.IsA(UsdGeom.Mesh)):
                #utils.setCollider(prim, approximationShape="convexDecomposition")
                utils.setCollider(prim, approximationShape="none")
                #print(prim)
        #for i in range(150):
            #the_world.step(render=True)
        
        stage = omni.usd.get_context().get_stage()
        label_to_object_number_map = {}
        curr_prim = stage.GetPrimAtPath("/World/objects")
     

        with open(self.data_root+f"/stage_{stage_ind}/"+f"stage_{stage_ind}_candidates_after_seal"+".pkl", "rb") as f:
            #print(f)
            candidates= pickle.load(f)
        f.close()

        draw = _debug_draw.acquire_debug_draw_interface()
        #result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Plane",prim_path="/World/Plane")

        add_cube(stage, "/World/Cube", offset=Gf.Vec3d(0, 0, 30))

        cube_prim_plane = stage.GetPrimAtPath("/World/Cube")
        xform = UsdGeom.Xformable(cube_prim_plane)
        #form.AddTranslateOp().Set(Gf.Vec3d(0.0,0.0,30.0))   
        xform.AddScaleOp().Set(Gf.Vec3f(100, 100, 0.1))

        utils.setCollider(cube_prim_plane, approximationShape="convexHull")

       # utils.setRigidBody(cube_prim_plane, "convexHull", False)
        #for i in range(50):
        #the_world.step(render=True)
        simulation_app.update()
        candidates_copy = copy.deepcopy(candidates)

        for i in candidates.keys():
            ##################################################################
            curr_prim = stage.GetPrimAtPath(f"/World/objects/object_{i-1}")
            imageable = UsdGeom.Imageable(curr_prim)
            time = Usd.TimeCode.Default() # The time at which we compute the bounding box
            bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
            centroid = bound.ComputeCentroid()
            utils.removeCollider(curr_prim)
          
            ray_directions = get_ray_directions(50000)
            distances = []  # Initialize an empty list to store distances

            for ray_dir in ray_directions:
                centroid = [centroid[0], centroid[1], centroid[2]]
                ray_dir = [ray_dir[0], ray_dir[1], ray_dir[2]]
                k, dis, pos = check_raycast(centroid, ray_dir, draw,False)
                if dis is not None:  # Only add valid distances
                    distances.append(dis)
                #print(k, dis, pos)

            # Compute the average distance
            if distances:  # Only if there are valid distances
                avg_distance = sum(distances) / len(distances)
                #normalized_distance = (avg_distance / 80) * 100  # Normalizing to the range 0-100
                print(f"Average free space for object {i-1}: {avg_distance}")

            ##################################################################
                 
            good_candidates_after_seal_T=candidates[i]["translation_after_seal_pass"]

            good_candidates_after_seal_R=candidates[i]["rotation_after_seal_pass"]

            ##################################################################

            #utils.setCollider(cube_prim_plane, approximationShape="convexHull")
            #utils.removeCollider(cube_prim_plane)


            utils.setCollider(curr_prim, approximationShape="none")

            
            stage = omni.usd.get_context().get_stage()
            stage.RemovePrim("/World/Cube")

            
            candidates_copy[i]["sampling_data"] = []
            candidates_copy[i]["free_space"]=avg_distance
            #the_world.step()
        
        for i in candidates.keys():
            good_candidates_after_seal_T=candidates[i]["translation_after_seal_pass"]

            good_candidates_after_seal_R=candidates[i]["rotation_after_seal_pass"]

            for suction_translation, suction_rotation_matrix in zip(good_candidates_after_seal_T, good_candidates_after_seal_R):
                # Initialize an empty list to store the sampled push directions
                # Generate the push samples
                push_samples_data = []
                grasp_samples_data = []
                #########################push_begin#####################################
                push_samples=6
                curr_prim = stage.GetPrimAtPath(f"/World/objects/object_{i-1}")
           
                for prim in Usd.PrimRange(curr_prim):
                    if (prim.IsA(UsdGeom.Mesh)):
                    #utils.setCollider(prim, approximationShape="convexDecomposition")
                    #utils.setCollider(prim, approximationShape="none")
                        utils.removeCollider(prim)
                utils.removeCollider(curr_prim)

                #the_world.step()
                the_world.step()
                for push_i in range(push_samples):
                    # Compute the rotation angle in radians
                    rotation_angle = 2 * np.pi * push_i/ push_samples  # This will range from 0 to 2*pi
                    push_origin,push_direction=sample_push(rotation_angle,suction_translation,suction_rotation_matrix,draw,the_world,False)
                    # Store push sample
                    maxDist=10
                    endPoint = [push_origin[0] + push_direction[0]*maxDist, push_origin[1] + push_direction[1]*maxDist, push_origin[2] + push_direction[2]*maxDist]
                    endPoint = [matrix.item() for matrix in endPoint]
                    
                    collision_distances=[]
                    #print(i)
                    for start_ray in good_candidates_after_seal_T:
                        #print(i)
                        _, push_dis_check, _=check_raycast(start_ray,push_direction,draw,False)
                        if push_dis_check!=None:
                            collision_distances.append(push_dis_check)
                        #for kkkkk in range(20):
                            #the_world.step()
                        #draw.clear_lines()
                    #push_dis_check = sum(collision_distances) / len(collision_distances)
                    if collision_distances==[]:
                        push_dis_check=22
                    else:
                        push_dis_check = min(collision_distances)

                    #print(push_dis_check)
                    if push_dis_check==None:
                        foward_score=1
                    elif push_dis_check>=20:
                        foward_score=1
                    else:
                        foward_score=0

                    push_samples_data.append({
                        "push_origin": push_origin,
                        "push_direction": push_direction,
                        "end_point": endPoint,
                        "forward_score": foward_score
                    })
                #########################push_end######################################
                
                #########################grasp_samples_begin######################################
                # Initialize an empty list to store the rotated darboux frames
                pinch_grasp_samples=36
                utils.setCollider(curr_prim, approximationShape="none")
                for prim in Usd.PrimRange(curr_prim):
                    if (prim.IsA(UsdGeom.Mesh)):
                        utils.setCollider(prim, approximationShape="convexDecomposition")
                    #utils.setCollider(prim, approximationShape="none")
                       # utils.removeCollider(prim)

                the_world.step()
                for pinch_grasp_samples_j in range(pinch_grasp_samples):
                    # Compute the rotation angle in radians
                    rotation_angle = 2 * np.pi * pinch_grasp_samples_j/ pinch_grasp_samples  # This will range from 0 to 2*pi
                    suction_translation, grasp_rot, collision_quality=sample_pinch_grasp(rotation_angle,suction_translation,suction_rotation_matrix,draw,the_world,False)
                    #sum_c += collision_quality
                    grasp_samples_data.append({
                    "suction_translation": suction_translation,
                    "grasp_rot": grasp_rot,
                    "collision_quality": collision_quality,
                    })
                #print(sum_c/pinch_grasp_samples)
                candidates_copy[i]["sampling_data"].append({
                    "suction_translation": suction_translation,
                    "suction_rotation_matrix": suction_rotation_matrix,
                    "push_samples": push_samples_data,
                    "grasp_samples": grasp_samples_data,
                })




                #########################grasp_samples_end######################################

                #print(candidates_copy[1]["sampling_data"][0]["push_samples"])######################
        
                #stage.GetPrimAtPath(f"/World/objects/object_{i-1}")
        
        #for prim in Usd.PrimRange(curr_prim):
           # if (prim.IsA(UsdGeom.Xform)):
                #utils.setCollider(prim, approximationShape="convexDecomposition")
        for i in candidates.keys():
            rigid_prim=stage.GetPrimAtPath(f"/World/objects/object_{i-1}")
            utils.setRigidBody(rigid_prim, "convexDecomposition", False)
                
                
                #print(prim)
        #prim_id = PhysicsSchemaTools.sdfPathToInt(f"/World/objects/object_{0}")
        #stage_cache = UsdUtils.StageCache.Get()
        #stage_id = stage_cache.GetId(stage).ToLongInt()
        #gripper_usd="/home/ubuntu/Documents/ur10_invisible_suction.usd"
        #new_position=np.array([-80,0,50])
        #robots=the_world.scene.add(UR10(
          #              prim_path=f"/World/Robot",
           #             name=f"ur10",
           #             usd_path= gripper_usd,
           #             position=new_position,
            #            attach_gripper=True,
            #            end_effector_prim_name="ee_link",
                        #gripper_usd= suction_usd,
             #       )) 

        #force = carb.Float3(1000.0, 1000.0, 1000.0)                                      
        #psi.apply_force_at_pos(stage_id, prim_id, force, carb._carb.Float3(location))

        #controllers = RMPFlowController(name="target_follower_controller", robot_articulation=robots, attach_gripper=True)
        #controllers=(PickPlaceController(name="pick_place_controller", gripper = robots.gripper, robot_articulation = robots))
                    
        #articulation_controllers=robots.get_articulation_controller()
        
        the_world.reset()
        #controllers.reset()
        the_world.play()


        
        for i in candidates.keys():
            curr_prim = stage.GetPrimAtPath(f"/World/objects/object_{i-1}")
            imageable = UsdGeom.Imageable(curr_prim)
            time = Usd.TimeCode.Default() # The time at which we compute the bounding box
            bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
            centroid_before = bound.ComputeCentroid()

            # If simulation is stopped, then exit.
            if the_world.is_stopped():
                break
            # If simulation is paused, then skip.
            if not the_world.is_playing():
               
                #controllers.reset()
                the_world.step(True)
                continue
            

            for suction_number in range(len(candidates[i]["translation_after_seal_pass"])):
                for push_number in range(len(candidates_copy[i]["sampling_data"][suction_number]["push_samples"])):
                    if candidates_copy[i]["sampling_data"][suction_number]["push_samples"][push_number]["forward_score"]==1:
                        suction_translation=candidates_copy[i]["sampling_data"][suction_number]["push_samples"][push_number]["push_origin"]
                        endPoint=candidates_copy[i]["sampling_data"][suction_number]["push_samples"][push_number]["end_point"]
                        push_direction=candidates_copy[i]["sampling_data"][suction_number]["push_samples"][push_number]["push_direction"]
                        suction_rotation_matrix= candidates_copy[i]["sampling_data"][suction_number]["suction_rotation_matrix"]
                        prev_distance = None  # Initialize a variable to store the previous distance outside the loop
                        while True:
                            a=Rz(np.pi)
                            R= np.matmul(suction_rotation_matrix, a)

                            T1 = np.eye(4)
                            T1[:3, :3] =suction_rotation_matrix
                            T1[0, 3] = suction_translation[0]
                            T1[1, 3] = suction_translation[1]
                            T1[2, 3] = suction_translation[2]
                            #print(new_task[f"{i}"]["prim_name"])
                            
                            #print(T1)
                            T2 = np.eye(4)
                            #T2[:3, :3] = R
                            T2[0, 3] = 0
                            T2[1, 3] = 0
                            T2[2, 3] = 0
                            
                            T3=np.matmul(T1, T2)
                            t=[T3[0, 3],T3[1, 3],T3[2, 3]]



                            T2[0, 3] = 5
                            T3=np.matmul(T1, T2)
                            t2=[T3[0, 3],T3[1, 3],T3[2, 3]]

                            
                            prim_id = PhysicsSchemaTools.sdfPathToInt(f"/World/objects/object_{i-1}")
                            stage_cache = UsdUtils.StageCache.Get()
                            stage = omni.usd.get_context().get_stage()

                            stage_id = stage_cache.GetId(stage).ToLongInt()


                            psi = get_physx_simulation_interface()
                            
                            #normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)

                            # Calculate the force vector
                            force_vector = 1000 * push_direction

                            force = carb.Float3(force_vector[0], force_vector[1], force_vector[2])
                            #print(force)
                            #print(push_direction)

                            psi.apply_force_at_pos(stage_id, prim_id, force, carb._carb.Float3(t))

                            #psi.apply_force_at_pos(stage_id, prim_id, force, carb._carb.Float3(location))
                            curr_prim = stage.GetPrimAtPath(f"/World/objects/object_{i-1}")
                            imageable = UsdGeom.Imageable(curr_prim)
                            time = Usd.TimeCode.Default() # The time at which we compute the bounding box
                            bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
                            centroid_after = bound.ComputeCentroid()

                            distance = np.linalg.norm(centroid_after - centroid_before)
                            if prev_distance is not None and np.abs(distance - prev_distance) < 0.0001:  # Adjust the threshold as needed
                                print("Push distance did not change significantly")
                                #the_world.reset()

                                #controllers.reset()
                                #the_world.reset()
                                print("done pushing")
                                print(stage_ind,i,suction_number,push_number)
                                #stage = Usd.Stage.Open(base_layer)
                                root_layer = stage.GetRootLayer()
                                #root_layer.subLayerPaths.append(sublayer)
                                root_layer.Export("/media/juncheng/ubuntu_data1/home/ubuntu/Downloads/amazon/after_push"+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}_{i}_{suction_number}_{push_number}.usd")
                                # Using reload here to clear the changes for the next loop
                                


                                #omni.usd.get_context().save_as_stage("/mnt/home/ubuntu/Downloads/amazon/after_push"+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}_{i}_{suction_number}_{push_number}.usd", None)
                                #omni.usd.get_context().reopen_stage()
                                
                                #stage.Reload()
                                the_world.reset()



                                break
                            

                            
                            if distance>= 10:
                                #the_world.reset()

                                #controllers.reset()
                                #the_world.reset()
                                print("done pushing")
                                print(stage_ind,i,suction_number,push_number)
                                #stage = Usd.Stage.Open(base_layer)
                                root_layer = stage.GetRootLayer()
                                #root_layer.subLayerPaths.append(sublayer)
                                root_layer.Export("/media/juncheng/ubuntu_data1/home/ubuntu/Downloads/amazon/after_push"+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}_{i}_{suction_number}_{push_number}.usd")
                                # Using reload here to clear the changes for the next loop
                                


                                #omni.usd.get_context().save_as_stage("/mnt/home/ubuntu/Downloads/amazon/after_push"+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}_{i}_{suction_number}_{push_number}.usd", None)
                                #omni.usd.get_context().reopen_stage()
                                
                                stage.Reload()
                                the_world.reset()
                                break
                            
                            prev_distance = distance




                           # Ensure the direction vector is normalized
                            
                                                        
                            
                            the_world.step()
        ##3#################candidates_copy[1]["sampling_data"][0]["push_samples"])######################
        
                   


        with open(self.data_root+f"/stage_{stage_ind}/"+f"stage_{stage_ind}_candidates_before_push"+".pkl", "wb") as f2:
                pickle.dump(candidates_copy, f2)
        f2.close()

        simulation_app.update()




        #simulation_app.close()
        #simulation_app=SimulationApp({"headless": False})
        return None

    def __len__(self):
        return len(self.stage_root)

#data_root="/home/juncheng/Downloads/revision_test"
#data_root="/home/juncheng/Downloads/iros"
data_root="/media/juncheng/ubuntu_data1/home/ubuntu/Downloads/amazon/data"
from torch.utils.data import DataLoader


stage_root = os.listdir(data_root)
len(stage_root)

exclusion_list = [217, 272, 284, 315, 368, 383, 413,418,421,429]


my_dataset = Dataset(data_root=data_root)

for i in range(422,len(stage_root)):
   # if i in exclusion_list:
        #gc.collect()
        my_dataset.__getitem__(i)
    
   
    #cProfile.run('my_dataset.__getitem__(i)')
    
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

###98 111 112 124 130 134 148 161 180 188 217 272 284 315 368 383 413











