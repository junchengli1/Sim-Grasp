
import os
import sys
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from models.SimGraspDataset import simgraspdata
from models.SimGraspNet import Sim_Grasp_Net
from torch.utils.data import DataLoader
import open3d as o3d
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from loss import get_loss
from pytorch_utils import BNMomentumScheduler
from torch.optim.lr_scheduler import StepLR
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="/media/juncheng/Disk4T1/Sim-Grasp/Sim_GraspNet/logs/log/sim_grasp_epoch50.tar")
parser.add_argument('--model_name', type=str, default='sim_grasp', help='Name of the model')
parser.add_argument('--log_dir', default='logs/log')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='30,34,38', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--resume', action='store_true', default=True, help='Whether to resume from checkpoint')
cfgs = parser.parse_args()
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass
def custom_collate_fn(batch):
    batch_point_clouds = [item['point_clouds'] for item in batch]
    batch_coors = [item['coors'] for item in batch]
    batch_feats = [item['feats'] for item in batch]
    batch_dense_scores = [item['pointwise_label'] for item in batch]
    batch_sparse_points = [item['sparse_points'] for item in batch]
    
    # Handle other fields as necessary. For variable-sized inputs, consider returning them as lists
    # or apply padding to standardize their sizes, depending on your model's requirements.
    
    return {
        'point_clouds': batch_point_clouds,
        'coors': batch_coors,
        'feats': batch_feats,
        'dense_scores': batch_dense_scores,
        'sparse_points': batch_sparse_points,
    }
def pad_collate(batch):
    # Find the maximum number of points in any sample in the batch
    max_points = max([sample['sparse_points'].shape[0] for sample in batch])
    max_N = max([sample['view_rotation_matrix'].shape[0] for sample in batch])

    # Pad each sample and convert to PyTorch tensor
    padded_batch = [{
        'point_clouds': torch.tensor(sample['point_clouds'], dtype=torch.float32),
        'coors': torch.tensor(sample['coors'], dtype=torch.float32),
        'feats': torch.tensor(sample['feats'], dtype=torch.float32),
        'pointwise_label': torch.tensor(sample['pointwise_label'], dtype=torch.float32),
        'sparse_points': torch.nn.functional.pad(torch.tensor(sample['sparse_points'], dtype=torch.float32), (0, 0, 0, max_points - sample['sparse_points'].shape[0]), 'constant', 0),
        'view_label': torch.nn.functional.pad(torch.tensor(sample['view_label'], dtype=torch.float32), (0, 0, 0, max_points - sample['view_label'].shape[0]), 'constant', 0),
        'view_rotation_matrix': torch.nn.functional.pad(torch.tensor(sample['view_rotation_matrix'], dtype=torch.float32), (0, 0, 0, 0, 0, 0, 0, max_N - sample['view_rotation_matrix'].shape[0]), 'constant', 0),
        'grasp_score': torch.nn.functional.pad(torch.tensor(sample['grasp_score'], dtype=torch.float32), (0, 0, 0, 0, 0, 0, 0, max_N - sample['grasp_score'].shape[0]), 'constant', 0),
        } 
        for sample in batch
        ]
        
    #Combine the dictionaries into a single dictionary with stacked tensors
    stacked_batch = {key: torch.stack([sample[key] for sample in padded_batch], dim=0) for key in padded_batch[0]}

    return stacked_batch

def pad_collate(batch):
    # Find the maximum number of points in any sample in the batch
    max_points = max([sample['sparse_points'].shape[0] for sample in batch])
    max_N = max([sample['view_rotation_matrix'].shape[0] for sample in batch])

    # Pad each sample and convert to PyTorch tensor
    padded_batch = [{
        'point_clouds': torch.tensor(sample['point_clouds'], dtype=torch.float32),
        'coors': torch.tensor(sample['coors'], dtype=torch.float32),
        'feats': torch.tensor(sample['feats'], dtype=torch.float32),
        'pointwise_label': torch.tensor(sample['pointwise_label'], dtype=torch.float32),
        'sparse_points': torch.nn.functional.pad(torch.tensor(sample['sparse_points'], dtype=torch.float32), (0, 0, 0, max_points - sample['sparse_points'].shape[0]), 'constant', 0),
        'view_label': torch.nn.functional.pad(torch.tensor(sample['view_label'], dtype=torch.float32), (0, 0, 0, max_points - sample['view_label'].shape[0]), 'constant', 0),
        'view_rotation_matrix': torch.nn.functional.pad(torch.tensor(sample['view_rotation_matrix'], dtype=torch.float32), (0, 0, 0, 0, 0, 0, 0, max_N - sample['view_rotation_matrix'].shape[0]), 'constant', 0),
        'grasp_score': torch.nn.functional.pad(torch.tensor(sample['grasp_score'], dtype=torch.float32), (0, 0, 0, 0, 0, 0, 0, max_N - sample['grasp_score'].shape[0]), 'constant', 0),
        } 
        for sample in batch
        ]
        
    #Combine the dictionaries into a single dictionary with stacked tensors
    stacked_batch = {key: torch.stack([sample[key] for sample in padded_batch], dim=0) for key in padded_batch[0]}

    return stacked_batch
def pad_new_collate(batch):
    # Find the maximum number of points in any sample in the batch
    max_N = max([sample['rotated_approach_directions'].shape[0] for sample in batch])

    # Pad each sample and convert to PyTorch tensor
    padded_batch = [{
        "augument_matrix":torch.tensor(sample['augument_matrix'], dtype=torch.float32),
        'point_clouds': torch.tensor(sample['point_clouds'], dtype=torch.float32),
        'coors': torch.tensor(sample['coors'], dtype=torch.float32),
        'feats': torch.tensor(sample['feats'], dtype=torch.float32),
        'sparse_points': torch.nn.functional.pad(torch.tensor(sample['sparse_points'], dtype=torch.float32), (0, 0, 0, max_N - sample['sparse_points'].shape[0]), 'constant', 0),
        'normalized_scores': torch.nn.functional.pad(torch.tensor(sample['normalized_scores'], dtype=torch.float32), (0, max_N - sample['normalized_scores'].shape[0]), 'constant', 0),
        'rotated_approach_directions': torch.nn.functional.pad(torch.tensor(sample['rotated_approach_directions'], dtype=torch.float32), (0, 0, 0, 0, 0, max_N - sample['rotated_approach_directions'].shape[0]), 'constant', 0),
        'normalized_view_score': torch.nn.functional.pad(torch.tensor(sample['normalized_view_score'], dtype=torch.float32), (0, 0, 0, max_N - sample['normalized_view_score'].shape[0]), 'constant', 0),
        'normalized_grasp_score': torch.nn.functional.pad(torch.tensor(sample['normalized_grasp_score'], dtype=torch.float32), (0, 0, 0, 0, 0, 0, 0, max_N - sample['normalized_grasp_score'].shape[0]), 'constant', 0),
        } 
        for sample in batch
        ]
        
    # Combine the dictionaries into a single dictionary with stacked tensors
    stacked_batch = {key: torch.stack([sample[key] for sample in padded_batch], dim=0) for key in padded_batch[0]}

    return stacked_batch
data_root= "/media/juncheng/Disk4T1/Sim-Grasp/single_pointcloud"
label_root="/media/juncheng/Disk4T1/Sim-Grasp/synthetic_data_grasp_cluster"
my_dataset = simgraspdata(data_root,label_root)

TRAIN_DATALOADER = DataLoader(my_dataset, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=my_worker_init_fn,collate_fn=pad_new_collate)
print('train dataloader length: ', len(TRAIN_DATALOADER))

net =  Sim_Grasp_Net(seed_feat_dim=256)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load the Adam optimizer
#optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
start_epoch = 0
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumSchedule


scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=start_epoch-1)


if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = cfgs.learning_rate
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr
# def get_current_lr(epoch):
#     lr = cfgs.learning_rate
#     lr = lr * (0.95 ** epoch)
#     return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
   
    net.train()
    batch_interval = 20
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        end_points = net(batch_data_label)
        loss, end_points = get_loss(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if (batch_idx+1) % 1 == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0
        


def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()

        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))
        #scheduler.step()

if __name__ == '__main__':
    train(start_epoch)