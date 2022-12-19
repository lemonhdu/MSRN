import random
import torch

import numpy as np

def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def init_seed(args):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_time_points(predicted):

    predicted = predicted.cpu().numpy()

    points = []

    predicted_length = len(predicted[0])

    state_index = 'idle'
    
    num_points = 0
    for i in range(predicted_length):

        if predicted[0][i] == 0 and predicted[0][i + 1] == 1 and state_index == 'idle':
            num_points += 1
            state_index = 'jump'
            
        elif predicted[0][i] == 1 and predicted[0][i+1] == 2 and (state_index == 'jump' or 'idle'):
            points.append(i+1)
            num_points += 1
            state_index = 'drop'

        elif predicted[0][i] == 2 and predicted[0][i+1] == 3 and state_index == 'drop':
            points.append(i+1)
            num_points += 1
            state_index = 'entering'

        elif predicted[0][i] == 3 and predicted[0][i+1] == 4 and state_index == 'entering':
            points.append(i+1)
            num_points += 1
            state_index = 'ending'
            break

    if points == []:
        return None, None

    else:
        points = np.array(points).reshape(1, -1)
        return points, num_points


def trans_time_labels(time_labels):

    num_labels = len(time_labels)

    points = []

    if num_labels == 3:
        index_start = 0
    elif num_labels == 4:
        index_start = 1

    for i in range(index_start, index_start+3):
        points.append(time_labels[i].item())

    points = np.array(points).reshape(1, 3)

    return points


def get_stage_features(i3d_feature, tcn_predict_time_points, time_point_num):

    i3d_feature = i3d_feature.squeeze(0).transpose(1, 0)

    if time_point_num == 3:
        # [jumping, dropping point]
        i3d_feature_jumping = i3d_feature[:tcn_predict_time_points[0][0]]

    elif time_point_num == 4:
        min_index = tcn_predict_time_points[0][0]-5 if tcn_predict_time_points[0][0]-5 >= 0 else 0
        # [dropping point - 5, dropping point]
        i3d_feature_jumping = i3d_feature[min_index:tcn_predict_time_points[0][0]]

    # [dropping point, entering point]
    i3d_feature_dropping = i3d_feature[tcn_predict_time_points[0][0]:tcn_predict_time_points[0][1]]
    # [entering point, ending point]
    i3d_feature_entering = i3d_feature[tcn_predict_time_points[0][1]:tcn_predict_time_points[0][2]]
    # [ending point, end]
    i3d_feature_ending = i3d_feature[tcn_predict_time_points[0][2]:]

    return i3d_feature_jumping, i3d_feature_dropping, i3d_feature_entering, i3d_feature_ending


