import numpy as np
import os
import glob
import pickle
import argparse
from pathlib import Path
base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='Dataset_generation')
parser.add_argument('--asset_path', type=str, default=(base_dir.parent / 'ShapeNet/*.usd').as_posix(), help='nvidia assets')
parser.add_argument('--asset_path1', type=str, default=(base_dir.parent / 'nvidia_assets/*.usd').as_posix(), help='shapenet (subset)')
parser.add_argument('--seg_dic_path', type=str, default=(base_dir.parent / "grasp_seg_dic.pkl").as_posix(), help='seg_dic path')

args = parser.parse_args()

screw_asset_paths=glob.glob(args.asset_path)
screw_asset_paths1=glob.glob(args.asset_path1)
#screw_asset_paths1=[]

asset_total=screw_asset_paths+screw_asset_paths1

seg_dic = {"ground": 0}
for i, path in enumerate(asset_total, start=1):
    tail = os.path.split(path)[1].lower()
    seg_dic[tail] = i

with open(args.seg_dic_path, "wb") as f:
    pickle.dump(seg_dic, f)

print(seg_dic)
