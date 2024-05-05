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

current_directory = os.path.abspath(os.path.dirname(__file__))
# Build the path to the desired directory
desired_directory = os.path.abspath(os.path.join(current_directory, "..", "Sim_GraspNet"))
# Insert the desired directory into sys.path
sys.path.insert(0, desired_directory)

from models.SimGraspNet_cluster import Sim_Grasp_Net
# Grounding DINO
from GroundingDINO.groundingdino.util.utils import  get_phrases_from_posmap
# segment anything
from segment_anything import (
    sam_model_registry,
    build_sam_hq,
    SamPredictor
)

parser = argparse.ArgumentParser("Sim-Grasp-Policy-Model", add_help=True)
parser.add_argument("--config", type=str, default=current_directory+"/config/GroundingDINO_SwinB.py", help="path to config file")
parser.add_argument("--grounded_checkpoint", type=str, default=current_directory+"/weights/groundingdino_swinb_tune.pth", help="path to checkpoint file")
parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
parser.add_argument("--sam_checkpoint", type=str, default="vit_h",required=False, help="path to sam checkpoint file")
parser.add_argument("--sam_hq_checkpoint", type=str, default=current_directory+"/weights/sam_hq_vit_h.pth", help="path to sam-hq checkpoint file")
parser.add_argument("--use_sam_hq",type=bool, default=True, help="using sam-hq for prediction")
parser.add_argument('--sim_grasp_model_name', type=str, default="/sim_grasp_gpus_single_epoch66.tar", help='model name')
parser.add_argument('--sim_grasp_model_path', type=str, default= desired_directory + '/weights', help='saved model path')

args = parser.parse_args()
config_file = args.config  # change the path of the model config file
grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
sam_version = args.sam_version
sam_checkpoint = args.sam_checkpoint
sam_hq_checkpoint = args.sam_hq_checkpoint
use_sam_hq = args.use_sam_hq




def sim_grasp_policy_model(device):

    ######################################initialize Sim-GraspNet################################################################3
    # Init the model
    model_path=args.sim_grasp_model_path+args.sim_grasp_model_name
    net = Sim_Grasp_Net(seed_feat_dim=256, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # Remove 'module.' prefix if model was trained using DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (model_path, start_epoch))
    # set model to eval mode
    net.eval()
    #####################################################################################################

    ############################### initialize SAM#######################################################
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    ##################################################################################################
    
    ############################### initialize DINO#######################################################
    dino_model = load_model(config_file, grounded_checkpoint, device=device)
    ######################################################################################################

    return net,predictor,dino_model
