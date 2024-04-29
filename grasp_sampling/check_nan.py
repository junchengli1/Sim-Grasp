#__author__ = 'Juncheng_Li'
#__contact__ = 'li3670@purdue.edu'

import glob
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_set_path', default='/media/juncheng/Disk4T1/Sim-Grasp/synthetic_data_grasp_test/', help='dataset path [default: choose the parent folder path of stage folder]')
parser.add_argument("--stage_range", type=int, nargs=2, default=[0, 500], help='range of stage IDs to check [default: 0 500]')
parser.add_argument("--gripper", default="Fetch", choices=["Fetch", "Robotiq"], help="choose parallel jaw gripper")

FLAGS = parser.parse_args()

DATA_ROOT = FLAGS.data_set_path
STAGE_RANGE = FLAGS.stage_range
GRIPPER = FLAGS.gripper

def check_good_candidates(DATA_ROOT, STAGE_RANGE, GRIPPER):
    for STAGE_ID in range(STAGE_RANGE[0], STAGE_RANGE[1] + 1):
        print("STAGE_ID:",STAGE_ID)
        if GRIPPER == "Fetch":
            candidate_simulation_path = DATA_ROOT + f"stage_{STAGE_ID}" + f"/stage_{STAGE_ID}_grasp_simulation_candidates_test.pkl"
        elif GRIPPER == "Robotiq":
            candidate_simulation_path = DATA_ROOT + f"stage_{STAGE_ID}" + f"/stage_{STAGE_ID}_grasp_simulation_candidates_roq.pkl"

        try:
            with open(candidate_simulation_path, 'rb') as f:
                candidate_simulation = pickle.load(f)

            total_good_candidates = 0
            for object_index in candidate_simulation.keys():
                good_candidates = [sample for sample in candidate_simulation[object_index]["grasp_samples"] if sample["collision_quality"] == 1 and sample["simulation_quality"] == 1]
                total_good_candidates += len(good_candidates)

            if total_good_candidates == 0:
                print(f'Stage {STAGE_ID} has all 0 good candidates')
        except FileNotFoundError:
            print(f'File not found for Stage {STAGE_ID}')

if __name__ == "__main__":
    check_good_candidates(DATA_ROOT, STAGE_RANGE, GRIPPER)
