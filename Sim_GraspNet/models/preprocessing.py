import numpy as np
import pickle
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle
from collections import defaultdict, Counter

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
    return views.astype(np.float32)


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
class SimGraspDataPreprocessor:
    def __init__(self, data_root, label_root, block_size=150, num_points=20000):
        self.data_root = data_root
        self.label_root = label_root
        self.block_size = block_size
        self.num_points = num_points
        self.views = generate_grasp_views(samples=300)  # Generate views
        #z_axis = np.array([0, 0, 1])
        #self.view_rot_matrices = rotation_matrices_from_vectors(self.views, z_axis)

    def preprocess_and_save_stage_data(self):
        for room_idx in tqdm(range(0, 500)):
            #points = np.load(self.data_root + f"/{room_idx}.npz", allow_pickle=True)['arr_0']
            candidate_simulation_path = self.label_root + f"/stage_{room_idx}" + f"/stage_{room_idx}_grasp_simulation_candidates.pkl"
            with open(candidate_simulation_path, 'rb') as f:
                candidate_simulation = pickle.load(f)

            unique_t_ori_points, normalized_scores, approach_directions, normalized_view_score, normalized_grasp_score = preprocess_grasp_data(candidate_simulation)

            # Create a dictionary to store all preprocessed data
            preprocessed_data = {
                'unique_t_ori_points': unique_t_ori_points,
                'normalized_scores': normalized_scores,
                'approach_directions': approach_directions,
                'normalized_view_score': normalized_view_score,
                'normalized_grasp_score': normalized_grasp_score
            }

            # Create a directory for the stage if it doesn't exist
            stage_dir = os.path.join(self.label_root, f"stage_{room_idx}")
            os.makedirs(stage_dir, exist_ok=True)

            # Save the dictionary as a pickle file
            with open(os.path.join(stage_dir, f'stage_{room_idx}_preprocessed_data.pkl'), 'wb') as f:
                pickle.dump(preprocessed_data, f)

# Example usage
data_root = 'path_to_your_data'
label_root = '/media/juncheng/Disk4T1/Sim-Grasp/synthetic_data_grasp_test'
preprocessor = SimGraspDataPreprocessor(data_root, label_root)
preprocessor.preprocess_and_save_stage_data()