# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import os
import sys
import open3d as o3d
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


def tensor_to_o3d_point_cloud(tensor):
    """Converts a PyTorch tensor to an Open3D point cloud."""
    #print("Tensor shape before reshape:", tensor.shape)
    points = tensor.cpu().numpy().reshape(-1, 3).astype(np.float64)  # Reshape and convert to float64
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd
def create_cylinder_mesh(radius, height, resolution=30):
    """
    Creates a mesh for a cylinder with the given radius and height.

    Parameters:
    - radius: The radius of the cylinder.
    - height: The height of the cylinder.
    - resolution: The number of points around the circumference.

    Returns:
    - A Open3D TriangleMesh object representing the cylinder.
    """
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    mesh.compute_vertex_normals()
    return mesh

def draw_cylinders(batch_idx,xyz, new_xyz, rot, radius, hmin, hmax):
    """
    Draws cylinders at specified positions with given orientations.

    Parameters:
    - xyz: The original point cloud (B, N, 3).
    - new_xyz: The centroids of the cylinders (B, npoint, 3).
    - rot: The rotation matrices for the cylinders (B, npoint, 3, 3).
    - radius: The radius of the cylinders.
    - hmin: The minimum height of the cylinders.
    - hmax: The maximum height of the cylinders.
    """
    new_xyz_pcd = tensor_to_o3d_point_cloud(new_xyz[batch_idx])
    new_xyz_pcd.paint_uniform_color([1, 0, 0])  # Red for centroids

    #pcds_to_draw = [new_xyz_pcd]
    pcds_to_draw = []
    #original_pcd = o3d.geometry.PointCloud()
    #original_pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy().reshape(-1, 3))
    pcds_to_draw.append(new_xyz_pcd)

   
    for j in range(new_xyz.shape[1]):
            cylinder_mesh = create_cylinder_mesh(radius, hmax - hmin)
            cylinder_mesh.rotate(rot[batch_idx, j].cpu().numpy(), center=np.array([0, 0, 0]).reshape(3, 1))

            cylinder_mesh.translate(new_xyz[batch_idx, j].cpu().numpy() - np.array([0, 0, (hmax - hmin) / 2]))
            pcds_to_draw.append(cylinder_mesh)
            o3d.visualization.draw_geometries(pcds_to_draw)

    # Convert the original point cloud to Open3D format and add it to the list
 
    # Draw all geometries
    #o3d.visualization.draw_geometries(pcds_to_draw)

class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


class CylinderQuery(Function):
    @staticmethod
    def forward(ctx, radius, hmin, hmax, nsample, xyz, new_xyz, rot):
        # type: (Any, float, float, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the cylinders
        hmin, hmax : float
            endpoints of cylinder height in x-rotation axis
        nsample : int
            maximum number of features in the cylinders
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the cylinder query
        rot: torch.Tensor
            (B, npoint, 9) flatten rotation matrices from
                           cylinder frame to world frame

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.cylinder_query(new_xyz, xyz, rot, radius, hmin, hmax, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None


cylinder_query = CylinderQuery.apply


class CylinderQueryAndGroup(nn.Module):
    """
    Groups with a cylinder query of radius and height

    Parameters
    ---------
    radius : float32
        Radius of cylinder
    hmin, hmax: float32
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        super(CylinderQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
       
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, rot, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        #print(xyz,new_xyz)
        idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        #print(idx,idx.shape)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        #for batch_idx in range(new_xyz.shape[0]):
            #draw_cylinders(batch_idx,xyz, new_xyz, rot, self.radius, self.hmin, self.hmax)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)


        
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous() # (B, npoint, nsample, 3)
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()


        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz


        # for batch_idx in range(new_xyz.shape[0]):
        #     new_xyz_pcd = tensor_to_o3d_point_cloud(new_xyz[batch_idx])
        #     new_xyz_pcd.paint_uniform_color([1, 0, 0])  # Red for centroids

        #     B, _, npoint, nsample = grouped_xyz.shape
        #     flattened_xyz = grouped_xyz.permute(0, 2, 3, 1).reshape(B, npoint * nsample, 3)
        #     pcds_to_draw = [new_xyz_pcd]
        #     first_stage= [new_xyz_pcd]
        #     for j in range(new_xyz.shape[1]):
        #         pcds_to_draw = [new_xyz_pcd]

        #         # Create and add the cylinder for the j-th point
        #         cylinder_mesh = create_cylinder_mesh(self.radius, self.hmax - self.hmin)
        #         cylinder_mesh.rotate(rot[batch_idx, j].cpu().numpy(), center=np.array([0, 0, 0]).reshape(3, 1))
        #         cylinder_mesh.translate(new_xyz[batch_idx, j].cpu().numpy() - np.array([0, 0, (self.hmax - self.hmin) / 2]))
        #         first_stage.append(cylinder_mesh)
        #         o3d.visualization.draw_geometries(first_stage)

        #         # Add the samples for the j-th point
        #         start_idx = j * nsample
        #         end_idx = (j + 1) * nsample
        #         samples_pcd = tensor_to_o3d_point_cloud(flattened_xyz[batch_idx, start_idx:end_idx])
        #         samples_pcd.paint_uniform_color([0, 0, 1])  # Blue for samples
        #         pcds_to_draw.append(samples_pcd)

        #         # Visualize the j-th point with its samples
        #         o3d.visualization.draw_geometries(pcds_to_draw)

        #         #draw_cylinders(batch_idx,xyz, new_xyz, rot, self.radius, self.hmin, self.hmax)

        #         # Visualize the additional point cloud
        #         #additional_pcd = tensor_to_o3d_point_cloud(additional_point_cloud[batch_idx])
        #         #additional_pcd.paint_uniform_color([0, 0, 1])  # Blue for the additional point cloud
        #         #pcds_to_draw.append(additional_pcd)

        #         # Visualize the grouped points
              

        # Convert tensors to Open3D point clouds
        # print(grouped_xyz.shape)
        # #grouped_xyz_permuted = grouped_xyz.permute(0, 2, 3, 1)  # Permute to (B, npoint, nsample, 3)
        # additional_point_cloud =xyz
        # for batch_idx in range(new_xyz.shape[0]):
        #         new_xyz_pcd = tensor_to_o3d_point_cloud(new_xyz[batch_idx])
        #         new_xyz_pcd.paint_uniform_color([1, 0, 0])  # Red for centroids

        #         pcds_to_draw = [new_xyz_pcd]

        #         # Visualize the additional point cloud
        #         #additional_pcd = tensor_to_o3d_point_cloud(additional_point_cloud[batch_idx])
        #         #additional_pcd.paint_uniform_color([0, 0, 1])  # Blue for the additional point cloud
        #         #pcds_to_draw.append(additional_pcd)

        #         # Visualize the grouped points
        #         B, _, npoint, nsample = grouped_xyz.shape
        #         flattened_xyz = grouped_xyz.permute(0, 2, 3, 1).reshape(B, npoint * nsample, 3)

        #         pcd = tensor_to_o3d_point_cloud(flattened_xyz[batch_idx])
        #         pcd.paint_uniform_color(np.random.rand(3))  # Random color for each group
        #         pcds_to_draw.append(pcd)

        #         o3d.visualization.draw_geometries(pcds_to_draw)
       
        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
