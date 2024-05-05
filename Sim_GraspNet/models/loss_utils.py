""" Tools for loss computation.
    Author: chenxi-wang
"""

import torch
import numpy as np

GRASP_MAX_WIDTH = 0.1
GRASPNESS_THRESHOLD = 0.1
NUM_VIEW = 300
NUM_ANGLE = 12
NUM_DEPTH = 7
M_POINT = 1024



def batch_viewpoint_params_to_matrix_data(batch_towards, batch_angle):
    """Transform approach vectors and in-plane rotation angles to rotation matrices."""
    # Normalize the approach vectors to get axis_z

    
    axis_z = batch_towards 
    
    # Construct the y-axis as a cross product of the z-axis and an arbitrary vector
    zeros = torch.zeros(axis_z.shape[0], dtype=axis_z.dtype, device=axis_z.device)
    ones = torch.ones(axis_z.shape[0], dtype=axis_z.dtype, device=axis_z.device)
    arbitrary_vector = torch.stack([zeros, zeros, ones], dim=-1)
    axis_y = torch.cross(axis_z, arbitrary_vector)
    
    # Handle the case where the cross product results in a zero vector
    mask_y = torch.norm(axis_y, dim=1) == 0
    axis_y[mask_y] = torch.tensor([0, 1, 0], dtype=axis_z.dtype, device=axis_z.device)
    
    # Normalize the y-axis
    axis_y = axis_y / torch.norm(axis_y, dim=1, keepdim=True)
    
    # Construct the x-axis as a cross product of the y-axis and the z-axis
    axis_x = torch.cross(axis_y, axis_z)
    
    # Normalize the x-axis
    axis_x = axis_x / torch.norm(axis_x, dim=1, keepdim=True)
    
    # Stack the axes to form the rotation matrices
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    
    # Construct the in-plane rotation matrix R1 around the z-axis
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)
    R1 = R1.reshape(-1, 3, 3)
    
    # Combine the rotations
    batch_matrix = torch.matmul(R2, R1)
    
    return batch_matrix
def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [torch.FloatTensor, (N,3)]
                points in original coordinates
            transform: [torch.FloatTensor, (3,3)/(3,4)/(4,4)]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [torch.FloatTensor, (N,3)]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = torch.matmul(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = cloud.new_ones(cloud.size(0), device=cloud.device).unsqueeze(-1)
        cloud_ = torch.cat([cloud, ones], dim=1)
        cloud_transformed = torch.matmul(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def generate_grasp_views(N=300, phi=(np.sqrt(5) - 1) / 2, center=np.zeros(3), r=1):
    """ View sampling on a unit sphere using Fibonacci lattices.
        Ref: https://arxiv.org/abs/0912.4540

        Input:
            N: [int]
                number of sampled views
            phi: [float]
                constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
            center: [np.ndarray, (3,), np.float32]
                sphere center
            r: [float]
                sphere radius

        Output:
            views: [torch.FloatTensor, (N,3)]
                sampled view coordinates
    """
    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi ** 2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi ** 2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    views = np.array(views)
    norms = np.linalg.norm(views, axis=1, keepdims=True)
    normalized_views = views / norms

    return torch.from_numpy(normalized_views.astype(np.float32))


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """ Transform approach vectors and in-plane rotation angles to rotation matrices.

        Input:
            batch_towards: [torch.FloatTensor, (N,3)]
                approach vectors in batch
            batch_angle: [torch.floatTensor, (N,)]
                in-plane rotation angles in batch
                
        Output:
            batch_matrix: [torch.floatTensor, (N,3,3)]
                rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = (torch.norm(axis_y, dim=-1) == 0)
    axis_y[mask_y, 1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Author: Charles R. Qi
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss

def batch_rot_matrix(batch_towards, batch_angle):
    # Normalize the approach vectors
    batch_towards = batch_towards / torch.norm(batch_towards, dim=1, keepdim=True)
    
    # Compute the rotation matrix to align the approach vector with the z-axis
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=batch_towards.device)
    axis = torch.cross(batch_towards, z_axis)
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    angle = torch.acos(torch.clamp(batch_towards[:, 2], -1.0, 1.0))  # Dot product with z-axis
    R_align = rotation_matrix_from_axis_and_angle(axis, angle)
    
    # Compute the rotation matrix for the rotation around the z-axis
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R_z = torch.stack([cos, -sin, torch.zeros_like(cos), 
                       sin, cos, torch.zeros_like(cos), 
                       torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)], dim=-1)
    R_z = R_z.reshape(-1, 3, 3)
    
    # Combine the rotations
    batch_matrix = torch.bmm(R_z, R_align)
    
    return batch_matrix

def rotation_matrix_from_axis_and_angle(axis, angle):
    # Convert axis-angle representation to rotation matrix
    a = torch.cos(angle / 2.0)
    b, c, d = -axis * torch.sin(angle / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return torch.stack([aa+bb-cc-dd, 2*(bc-ad), 2*(bd+ac),
                        2*(bc+ad), aa+cc-bb-dd, 2*(cd-ab),
                        2*(bd-ac), 2*(cd+ab), aa+dd-bb-cc], dim=-1).reshape(-1, 3, 3)