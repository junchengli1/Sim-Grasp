import argparse
import os
import copy
import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

from sim_grasp_policy_utils import *
from sim_grasp_policy_model import sim_grasp_policy_model
from models.SimGraspNet_cluster import Sim_Grasp_Net, pred_decode
import open3d as o3d
import argparse
from inference_utils import plot_gripper,sort_grasps_by_score,grasp_nms_with_object_id,CollisionDetector
#matplotlib.use('Agg')
# Get the directory of the current script
current_directory = os.path.abspath(os.path.dirname(__file__))

# Grounding DINO
from GroundingDINO.groundingdino.util.utils import  get_phrases_from_posmap

#################################################################################################

parser = argparse.ArgumentParser("Sim-Grasp-Policy Demo", add_help=True)
#use all objects prompt if you want to pick up all objects and select the best to grasp#
#use one object prompt if you want to pick up one object#
parser.add_argument("--text_prompt", type=str, default="banana", help="text prompt picking up *****")
parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
parser.add_argument("--text_threshold", type=float, default=0.4, help="text threshold")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--demo_path", type=str, default=current_directory+"/rgb_pcd_640", help="device")
parser.add_argument("--demo_number", type=str, default="3_5")
#if suction confident score is lower than this threshold, the suction pose will not be considered as good pose#
parser.add_argument('--voxel_size', type=float, default=1, help='Voxel Size for sparse convolution')
parser.add_argument('--td', type=float, default=0.1, help='nms close point')

args = parser.parse_args()

text_prompt = args.text_prompt
box_threshold = args.box_threshold
text_threshold = args.text_threshold
device = args.device
demo_path=args.demo_path
demo_number=args.demo_number
td_threshold=args.td

def get_grounding_output2(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Filter output based on both box_threshold and text_threshold
    filt_mask = (logits.max(dim=1)[0] > box_threshold) & (logits > text_threshold).any(dim=1)
    logits_filt = logits[filt_mask]  # num_filt, 256
    boxes_filt = boxes[filt_mask]  # num_filt, 4

    # Get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # Build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def get_mask_dino_sam(predictor,model,image_pil,image):

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()


    widths = boxes_filt[:, 2] - boxes_filt[:, 0]
    heights = boxes_filt[:, 3] - boxes_filt[:, 1]

    # Define maximum width and height threshold
    #print("Widths:", widths)
    #print("Heights:", heights)
    max_width_threshold = 350
    max_height_threshold = 250

    # Create a mask for boxes within the thresholds
    valid_boxes_indices = torch.where((widths <= max_width_threshold) & (heights <= max_height_threshold))[0]

    # Filter the boxes using the mask
    filtered_boxes = boxes_filt[valid_boxes_indices]
    #filtered_boxes=[boxes_filt[i] for i in valid_boxes_indices]

    # Convert boolean mask to indices
    filtered_pred_phrases = [pred_phrases[i] for i in valid_boxes_indices]

    print(filtered_pred_phrases)
    transformed_boxes = predictor.transform.apply_boxes_torch(filtered_boxes, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    # draw output image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(filtered_boxes, filtered_pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)

    # plt.axis('off')
    # #plt.show()

    return masks

def remove_points_from_pointcloud(base_pcd, points_to_remove):
    """
    Removes points from base_pcd that are closest to points_to_remove.
    
    Args:
    - base_pcd (o3d.geometry.PointCloud): The point cloud from which points are to be removed.
    - points_to_remove (o3d.geometry.PointCloud): The point cloud containing points to be removed from base_pcd.
    
    Returns:
    - o3d.geometry.PointCloud: The modified point cloud.
    """
    
    # Build a KDTree for the base point cloud
    kdtree = o3d.geometry.KDTreeFlann(base_pcd)
    
    # For each point in points_to_remove, find the closest point in base_pcd
    indices_to_remove = []
    for point in np.asarray(points_to_remove.points):
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        indices_to_remove.append(idx[0])
    
    # Remove the identified indices from base_pcd
    base_points = np.asarray(base_pcd.points)
    base_colors = np.asarray(base_pcd.colors)
    mask = np.ones(base_points.shape[0], dtype=bool)
    mask[indices_to_remove] = False
    base_pcd.points = o3d.utility.Vector3dVector(base_points[mask])
    if base_colors.size > 0:
        base_pcd.colors = o3d.utility.Vector3dVector(base_colors[mask])
    
    return base_pcd

def preprocess_data(pcd, segmentation):
    """Preprocess point cloud and segmentation data."""
    points_read = np.zeros((len(pcd.points), 6))
    points_read[:, 0:3] = np.array(pcd.points)
    points_read[:, 3:6] = np.array(pcd.normals)
    point_idxs = np.where(segmentation != 1)[0]
    points_ori = copy.deepcopy(points_read[point_idxs, 0:3])
    points_filter = points_read[point_idxs]
    #voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, 1)
    # if len(points_filter) < 5120:
    #         # Calculate the number of points needed
    #         num_points_needed = 5120 - len(points_filter)           
    #         zero_padding = np.zeros((num_points_needed, points_filter.shape[1]))
    #         points_filter = np.vstack((points_filter, zero_padding))

    return points_filter, np.array(pcd.points), point_idxs

def evaluate_suction_model(points,point_idxs, grasp_model):
    points[:, 0:3] = pc_normalize(points[:, 0:3])

    points=torch.from_numpy(points).float(). unsqueeze(0)
    
    pc = points.cuda()

    _,output_score = grasp_model(pc, None)
    camera_resolution = (720, 1280)

    camera_res_idx = np.unravel_index(point_idxs, camera_resolution)
    camera_res_idx = np.array(camera_res_idx).T
    score_map = np.zeros(camera_resolution)
    for idx, score in zip(camera_res_idx, output_score):
        score_map[idx[0], idx[1]] = score
    return score_map,camera_res_idx

def create_masked_score_map(instance_mask):
    instance_mask = instance_mask.cpu().numpy()
    # Convert the instance_mask to binary format
    binary_mask = (instance_mask > 0).astype(np.float32)
    binary_mask=np.squeeze(binary_mask)
    #print(binary_mask.shape)
    centroid = calculate_centroid(binary_mask)
    dimensions = binary_mask.shape
    decay_factor=0.1
    heatmap = create_heatmap_centroid(dimensions, centroid, decay_factor)       
    masked_score_map = heatmap*score_map* binary_mask
    return masked_score_map

def get_smoothed_masked_score_map(masked_score_map):
    """Smooth the masked score map (if required)."""
    # smoothed = gaussian_filter(masked_score_map, sigma) # Uncomment if you want to use Gaussian filter
    return np.squeeze(masked_score_map)

def get_segmented_point_cloud_and_scores(points_ori, camera_res_idx, smoothed_masked_score_map, image_pil):
    """Retrieve segmented point cloud and corresponding scores."""
    smoothed_predicted_scores_instance = smoothed_masked_score_map[camera_res_idx[:, 0], camera_res_idx[:, 1]]
    rgb_colors = np.array(image_pil)[camera_res_idx[:, 0], camera_res_idx[:, 1]]
    non_zero_indices = np.nonzero(smoothed_predicted_scores_instance)[0]
    segmented_point_cloud = points_ori[non_zero_indices]
    return segmented_point_cloud, smoothed_predicted_scores_instance, smoothed_predicted_scores_instance[non_zero_indices], rgb_colors[non_zero_indices]

def visualize_segmented_points(segmented_point_cloud, normalized_scores, rgb_colors):
    """Visualize segmented points using a colormap."""
    cmap = matplotlib.cm.get_cmap('plasma')
    score_colors = cmap(normalized_scores)
    alpha = 0.7
    blended_colors = alpha * score_colors[:, :3] + (1 - alpha) * rgb_colors / 255.0
    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(np.array(segmented_point_cloud))
    segmented_pcd.colors = o3d.utility.Vector3dVector(blended_colors[:, :3])
    return segmented_pcd

def calculate_collision_free_suction_candidates(suction_confident_threshold,points_ori, kdtree, normals, smoothed_predicted_scores_instance, sort_idx, voxel_grid, display):
    for idx in sort_idx:
        t_ori, t_translate, Rotation_mat, normal_vector, _ = compute_darboux_frame(
            points_ori, idx, kdtree, normals)
        collision = check_collision_with_suction_gripper_using_voxelgrid(
            voxel_grid, t_translate, normal_vector)
        # If no collision, append to lists and break out of the loop to consider this as top-1 point
        if not collision:
            if smoothed_predicted_scores_instance[idx]>suction_confident_threshold:
                print("confident score:", smoothed_predicted_scores_instance[idx])
                confident=smoothed_predicted_scores_instance[idx]
                mesh = create_mesh_cylinder_detection_based_on_alpha(Rotation_mat, t_ori, confident,radius=1.5)
                display.append(mesh)
            break
    return display

def process_masks(suction_confident_threshold,masks, normals, camera_res_idx, points_ori, kdtree, voxel_grid, image_pil, global_min_score, global_max_score):
    """Process masks to obtain visualizations and top suction points."""
    display = []
    combined_pcd = o3d.geometry.PointCloud()
    for instance_mask in masks:
        masked_score_map=create_masked_score_map(instance_mask)
        
        smoothed_masked_score_map = get_smoothed_masked_score_map(masked_score_map)

        segmented_point_cloud, smoothed_predicted_scores_instance,segmented_scores, rgb_colors = get_segmented_point_cloud_and_scores(
            points_ori, camera_res_idx, smoothed_masked_score_map, image_pil)

        normalized_scores = (segmented_scores - global_min_score) / (global_max_score - global_min_score)
        segmented_pcd = visualize_segmented_points(segmented_point_cloud, normalized_scores, rgb_colors)
        combined_pcd += segmented_pcd

        sort_idx = np.argsort(smoothed_predicted_scores_instance)
        sort_idx = sort_idx[::-1]  # Sort in descending order
    
        display=calculate_collision_free_suction_candidates(suction_confident_threshold,points_ori, kdtree, normals, smoothed_predicted_scores_instance, sort_idx, voxel_grid, display)


    return display, combined_pcd

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def get_and_process_data_sim(points_seg):
    # load data
    
    # Downsample the point cloud to 20,000 points
    #downsampled_cloud = cloud.uniform_down_sample(int(len(cloud.points) / 40000))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_seg[:, 0:3].astype(np.float32))
    cloud.normals = o3d.utility.Vector3dVector(points_seg[:, 3:6].astype(np.float32))
   # cloud.colors = o3d.utility.Vector3dVector(points_seg[:, 6:9].astype(np.float32))

    downsampled_cloud=cloud
    # Convert back to NumPy array

    pc,centroid,m=pc_normalize(np.asarray(downsampled_cloud.points))
    N = pc.shape[0]  # Number of points in the point cloud

    # Initialize cloud_sampled with shape (N, 6)
    cloud_sampled = np.zeros((N, 6))

    cloud_sampled[:,0:3] = pc

    cloud_sampled[:,3:6]=np.asarray(downsampled_cloud.normals)

    end_points = dict()

    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cloud_sampled = cloud_sampled.to(device)

    end_points['point_clouds'] = cloud_sampled
    end_points['centroid']=centroid
    end_points['m']=m

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return gg_array

def collision_detection(cloud,gg_array):
    mfcdetector = CollisionDetector(cloud,voxel_size=args.voxel_size)
    collision_mask = mfcdetector.detect(gg_array)
    #gg=mfcdetector.detect2(gg)
    gg_array_collision_free = gg_array[~collision_mask]

    #gg = gg[~collision_mask]
    return gg_array_collision_free

def vis_grasps(gg_array_collision_free, cloud):
    # Filter out grasps with obj_id == -1
    valid_grasps = gg_array_collision_free[gg_array_collision_free[:, -1] != -1]

    keep_indices = grasp_nms_with_object_id(valid_grasps, thd=td_threshold, tha=np.deg2rad(30))
    nms_grasps = valid_grasps[keep_indices]
    sorted_grasps = sort_grasps_by_score(nms_grasps)
    hand_mesh = o3d.io.read_triangle_mesh("/home/juncheng/Documents/symbol2.ply")
    display_top_grasps(sorted_grasps, cloud, top_percent=50, hand_mesh=hand_mesh, top1_only=False)

def display_top_grasps(sorted_grasps, cloud, top_percent, hand_mesh,top1_only=False):
    # Calculate the number of grasps to display
    num_grasps = len(sorted_grasps)
    if top1_only:
        num_display_grasps = 1
    else:
        num_display_grasps = int(num_grasps * top_percent / 100)
        num_display_grasps = max(1, num_display_grasps)  # Display at least one grasp

    # Plot the top grasps
    gripper_meshes = []
    for grasp in sorted_grasps[:num_display_grasps]:
        score = grasp[0]
        depth = grasp[1]
        rot = grasp[2:11].reshape(3, 3)
        center = grasp[11:14]
        gripper_mesh = plot_gripper(hand_mesh,center, rot, depth, score)
        gripper_meshes.append(gripper_mesh)

    # Create a visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud and gripper meshes to the visualization
    
    vis.add_geometry(cloud)
    for gripper_mesh in gripper_meshes:
        vis.add_geometry(gripper_mesh)

    # Run the visualization
    vis.run()
    vis.destroy_window()

def create_point_indices(camera_resolution):
    height, width = camera_resolution
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    point_idxs = np.stack([x.ravel(), y.ravel()], axis=-1)
    return point_idxs

def segment_point_cloud(point_cloud, masks, point_idxs,points_ori):
    """
    Segment the point cloud using the instance masks from the image.
    Returns a list of segmented point clouds and their corresponding object IDs.
    """
    segmented_point_clouds = []
    obj_ids = []

    for idx, mask in enumerate(masks):
        # Convert the mask from torch tensor to numpy array and squeeze the single channel dimension
        mask_np = mask.squeeze(0).cpu().numpy()

        # Convert image mask to point cloud mask
        pc_mask = mask_np[point_idxs[:, 1], point_idxs[:, 0]]

        # Extract points corresponding to the mask
        segmented_points = points_ori[pc_mask]

        # Create a new point cloud for the segmented points
        segmented_pc = o3d.geometry.PointCloud()
        segmented_pc.points = o3d.utility.Vector3dVector(segmented_points)

        segmented_point_clouds.append(segmented_pc)
        obj_ids.append(idx + 1)  # Assign a unique object ID (starting from 1)

    return segmented_point_clouds, obj_ids

def visualize_segmented_point_clouds(segmented_point_clouds, obj_ids):
    """
    Visualize segmented point clouds with random colors for different object IDs.
    """
    # Create a dictionary to store a unique color for each object ID
    color_dict = {}

    # Assign a random color to each segmented point cloud based on its object ID
    for segmented_pc, obj_id in zip(segmented_point_clouds, obj_ids):
        if obj_id not in color_dict:
            # Generate a random color and store it in the dictionary
            color_dict[obj_id] = np.random.rand(3)
        segmented_pc.paint_uniform_color(color_dict[obj_id])

    # Visualize all segmented point clouds together
    o3d.visualization.draw_geometries(segmented_point_clouds)

def mark_grasp_obj_ids(grasps, segmented_point_clouds, obj_ids):
    """
    Mark the obj_ids in grasps based on the grasp_center and whether it is contained in the segmented point clouds.
    """
    for grasp in grasps:
        grasp_center = grasp[11:14]  # Extract the grasp center

        # Check if the grasp center is in any of the segmented point clouds
        for seg_pc, obj_id in zip(segmented_point_clouds, obj_ids):
            if seg_pc.has_points():
                # Convert the points of the segmented point cloud to a NumPy array
                seg_points = np.asarray(seg_pc.points)

                # Check if grasp center is one of the points in the segmented point cloud
                if np.any(np.all(np.isclose(seg_points, grasp_center, atol=1e-6), axis=1)):
                    grasp[-1] = obj_id  # Update the obj_id for the grasp
                    break
        else:
            grasp[-1] = -1  # Mark as -1 if the grasp center is not in any segmented point cloud

    return grasps

if __name__ == "__main__":

    grasp_model,predictor,dino_model=sim_grasp_policy_model(device)

    image_path=demo_path+f"/{demo_number}"+".png"
    pcl_path=demo_path+f"/{demo_number}"+".pcd"
    segmentation_path=demo_path+f"/{demo_number}"+".npz"

    point_cloud, image_pil, image, segmentation = load_data(image_path, pcl_path, segmentation_path)
    points_filtered, points_ori, _ = preprocess_data(point_cloud, segmentation)

    #with torch.no_grad():

    end_points, cloud = get_and_process_data_sim(points_filtered)

    gg_array = get_grasps(grasp_model, end_points)

    gg_array_collision_free = collision_detection(points_filtered[:,0:3],gg_array)
    
    masks=get_mask_dino_sam(predictor,dino_model,image_pil,image)
    
    camera_resolution = (720, 1280)

    point_idxs = create_point_indices(camera_resolution)

    segmented_point_clouds, obj_ids = segment_point_cloud(point_cloud, masks, point_idxs,points_ori)

    #visualize_segmented_point_clouds(segmented_point_clouds, obj_ids)

    marked_grasps = mark_grasp_obj_ids(gg_array_collision_free, segmented_point_clouds, obj_ids)

    
    vis_grasps(marked_grasps, point_cloud)


    if masks==None:
             print("no target detected")
    else:    
        print(masks.shape)
