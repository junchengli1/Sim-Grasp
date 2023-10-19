#__author__ = 'Juncheng_Li'
#__contact__ = 'li3670@purdue.edu'

import open3d as o3d
import glob
import pickle
import numpy as np
import sys
import argparse

import copy


parser = argparse.ArgumentParser()
parser.add_argument('--data_set_path', default='/media/juncheng/Disk4T/Sim-Grasp/synthetic_data/', help='dataset path [default: choose the parent folder path of stage folder]')
parser.add_argument('--gripper_symbol_path', default="/home/juncheng/Documents/symbol2.ply", help='choose the path of symbol gripper [default: Fetch wireframe')

parser.add_argument("--stage_ID", type=int, default=0, help='stage ID number [default: 0]')

parser.add_argument("--mode", default="collision", choices=["collision","simulation"],help="choose the type of check to visualize")
parser.add_argument("--ground", default=True, choices=[True,False],help="whether to include ground plane")
parser.add_argument("--gripper", default="Fetch",choices=["Fetch","Robotiq"],help="choose parallel jaw gripper")



FLAGS = parser.parse_args()

DATA_ROOT = FLAGS.data_set_path
STAGE_ID=FLAGS.stage_ID
MODE=FLAGS.mode
GROUND_PLANE=FLAGS.ground
GRIPPER=FLAGS.gripper
SYMBOL=FLAGS.gripper_symbol_path

def draw(DATA_ROOT,STAGE_ID,MODE,GROUND_PLANE,GRIPPER,SYMBOL):
    display=[]
    pcd_list=glob.glob(DATA_ROOT+ f"stage_{STAGE_ID}"+"/**/*.pcd", recursive=True)
    if GROUND_PLANE==False:
        pcd_list.remove(DATA_ROOT+ f"stage_{STAGE_ID}"+"/ground.pcd")
    if GRIPPER=="Fetch":
        candidate_overlap_path=DATA_ROOT+f"stage_{STAGE_ID}"+ f"/stage_{STAGE_ID}_grasp_candidates_after_overlap.pkl"
        candidate_simulation_path=DATA_ROOT+f"stage_{STAGE_ID}"+ f"/stage_{STAGE_ID}_grasp_simulation_candidates.pkl"
    elif GRIPPER=="Robotiq":
        candidate_seal_path=DATA_ROOT+f"stage_{STAGE_ID}"+ f"/stage_{STAGE_ID}_grasp_candidates_after_overlap_roq.pkl"
        candidate_simulation_path=DATA_ROOT+f"stage_{STAGE_ID}"+ f"/stage_{STAGE_ID}_grasp_simulation_candidates_roq.pkl"

    #with open(candidate_overlap_path, 'rb') as f:
        #candidate_overlap= pickle.load(f)
    
    
    hand_mesh=o3d.io.read_triangle_mesh(SYMBOL)


    if MODE=="collision":
        with open(candidate_overlap_path, 'rb') as f:
            candidate_overlap= pickle.load(f)
        for object_index in candidate_overlap.keys():
            grasp_samples=candidate_overlap[object_index]["grasp_samples"]
            good_candidates = [sample for sample in candidate_overlap[object_index]["grasp_samples"] if sample["collision_quality"]== 1]

            print(len(good_candidates))
            for i in range(len(grasp_samples)):

                suction_translation=grasp_samples[i]["suction_translation"]
                suction_rotation_matrix=grasp_samples[i]["suction_rotation_matrix"]
                collision=True
                T = np.eye(4)
                T[:3, :3] = suction_rotation_matrix
                T[0, 3] = suction_translation[0]
                T[1, 3] = suction_translation[1]
                T[2, 3] = suction_translation[2]
                if grasp_samples[i]["collision_quality"]==1:
                    hand_mesh.paint_uniform_color([0, 1, 0])
                else:

                    continue
                mesh_t = copy.deepcopy(hand_mesh).transform(T)
                display.append(mesh_t)

                


    if MODE=="simulation":
        with open(candidate_simulation_path, 'rb') as f:
            candidate_simulation= pickle.load(f)
        for object_index in candidate_simulation.keys():
            grasp_samples=candidate_simulation[object_index]["grasp_samples"]
            
            bad_candidates = [sample for sample in candidate_simulation[object_index]["grasp_samples"] if sample["collision_quality"]== 1 and sample["simulation_quality"] == 0]
            good_candidates = [sample for sample in candidate_simulation[object_index]["grasp_samples"] if sample["collision_quality"]== 1 and sample["simulation_quality"] == 1]

            
            print(len(good_candidates))
            for i in range(len(good_candidates)):

                suction_translation=good_candidates[i]["suction_translation"]
                suction_rotation_matrix=good_candidates[i]["suction_rotation_matrix"]
                collision=True
                T = np.eye(4)
                T[:3, :3] = suction_rotation_matrix
                T[0, 3] = suction_translation[0]
                T[1, 3] = suction_translation[1]
                T[2, 3] = suction_translation[2]
                hand_mesh.paint_uniform_color([0, 1, 0])
              
                mesh_t = copy.deepcopy(hand_mesh).transform(T)
                display.append(mesh_t)

            for i in range(len(bad_candidates)):

                suction_translation=bad_candidates[i]["suction_translation"]
                suction_rotation_matrix=bad_candidates[i]["suction_rotation_matrix"]
                collision=True
                T = np.eye(4)
                T[:3, :3] = suction_rotation_matrix
                T[0, 3] = suction_translation[0]
                T[1, 3] = suction_translation[1]
                T[2, 3] = suction_translation[2]
                hand_mesh.paint_uniform_color([1, 0, 0])
              
                #mesh_t = copy.deepcopy(hand_mesh).transform(T)
                #display.append(mesh_t)

    for i in pcd_list:

        pcd = o3d.io.read_point_cloud(i)
        display.append(pcd)

    o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)



if __name__ == "__main__":
    draw(DATA_ROOT,STAGE_ID,MODE,GROUND_PLANE,GRIPPER,SYMBOL)










