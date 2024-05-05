import torch.nn as nn
import torch


def get_loss(end_points):
    graspness_loss, end_points = compute_graspness_loss(end_points)
    view_loss, end_points = compute_view_graspness_loss(end_points)
    score_loss, end_points = compute_score_loss(end_points)
    loss = 45 * graspness_loss + 95 * view_loss + 70 * score_loss

    end_points['loss/overall_loss'] = loss
    return loss, end_points

def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')

    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['pointwise_label'].unsqueeze(-1)
    loss = criterion(graspness_score, graspness_label)
    loss = loss.mean()
    graspness_score_c = graspness_score.detach().clone()
    graspness_label_c = graspness_label.detach().clone()
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    end_points['stage1_graspness_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points



def compute_view_graspness_loss(end_points):
    criterion = nn.BCEWithLogitsLoss()
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points




def compute_score_loss(end_points):
    criterion = nn.BCEWithLogitsLoss()
    grasp_score_pred = end_points['grasp_score_pred']  # B , num_seed , num_angle , num_depth
    grasp_score_label = end_points['batch_grasp_score']  # B , num_seed , num_angle, num_depth

    # Ensure labels are float type for BCEWithLogitsLoss
    grasp_score_label = grasp_score_label.float()

    # Reshape predictions and labels to match the expected input shape for BCEWithLogitsLoss
    grasp_score_pred_reshaped = grasp_score_pred.view(grasp_score_pred.size(0), grasp_score_pred.size(1), -1)
    grasp_score_label_reshaped = grasp_score_label.view(grasp_score_label.size(0), grasp_score_label.size(1), -1)


    loss = criterion(grasp_score_pred_reshaped, grasp_score_label_reshaped)
    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points
