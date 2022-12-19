import os

import torch
from scipy import stats
from torch import optim

from config import get_parser
from models.mstcn import MultiStageModel, FeatureLink
from models.fc import FullConnectNet1, FullConnectNet2

from tqdm import tqdm
import torch.nn as nn

from params import *
from dataset import VideoDataset
from utils import *



def get_dataloaders(args):

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn
                                                       )

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn
                                                      )
    return dataloaders


def model_param_init(model):

    for param in model.parameters():
        nn.init.uniform_(param, a=0, b=0.05)


def get_models(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    mstcn_model = MultiStageModel(num_stages=tcn_num_stages,
                            num_layers=tcn_num_layers,
                            num_f_maps=tcn_num_f_maps,
                            dim=tcn_features_dim,
                            num_classes=tcn_num_classes).cuda()

    mstcn_model.load_state_dict(torch.load(mstcn_pretrained_path)['tcn_model'])

    feature_link = FeatureLink(input_channel=i3d_feature_dim, output_channel=i3d_feature_dim).cuda()

    # feature_link.load_state_dict(torch.load(best_base_full_connect_layer_dir)['feature_link'])

    full_connect1 = FullConnectNet1(feature_dim=i3d_feature_dim, output_dim=1).cuda()

    # full_connect1.load_state_dict(torch.load(best_base_full_connect_layer_dir)['full_connect1_model'])

    model_param_init(full_connect1)

    full_connect2 = FullConnectNet2(num_scores=4, num_output=1).cuda()

    # full_connect2.load_state_dict(torch.load(best_base_full_connect_layer_dir)['full_connect2_model'])

    model_param_init(full_connect2)

    best_spearman_c = 0  #torch.load(best_base_full_connect_layer_dir)['best_coefficient']

    return mstcn_model, feature_link, full_connect1, full_connect2, best_spearman_c


def compute_loss_score(predict_score, label_score):

    criterion_mse = nn.MSELoss(reduction='none')

    loss = criterion_mse(predict_score, label_score).float().cuda()

    return loss


def train_models(dataloaders, mstcn_model, feature_link, full_connect1, full_connect2, history_spearman_c, args):

    optimizer = optim.Adam([*feature_link.parameters()]+[*full_connect1.parameters()]+[*full_connect2.parameters()],
                           lr=args.lr)

    best_src = history_spearman_c
    for epoch in range(args.num_epochs):

        for split in ['train', 'test']:

            if split == 'train':
                full_connect1.train()
                full_connect2.train()
                feature_link.train()
                torch.set_grad_enabled(True)
            else:
                full_connect1.eval()
                full_connect2.eval()
                feature_link.eval()
                mstcn_model.eval()
                torch.set_grad_enabled(False)

            epoch_loss = 0

            empty_points_num = 0

            truth_score = []

            predict_score = []

            for data in tqdm(dataloaders[split]):

                label_final_score = data['final_score'].cuda()

                difficulty_level = data['difficulty'].cuda()

                label_execute_score = (30-label_final_score/difficulty_level).cuda()

                i3d_feature = data['video'].cuda()

                # get tcn predicted points
                num_feature = len(i3d_feature[0][0])
                mask = torch.ones(1, tcn_num_classes, num_feature, dtype=torch.float).cuda()

                tcn_predict_time = mstcn_model(i3d_feature, mask)

                _, predicted = torch.max(tcn_predict_time[-1].data, 1)
                tcn_predict_time_points, predicted_points_num = get_time_points(predicted)
                if tcn_predict_time_points is None:
                    empty_points_num = empty_points_num+1
                    continue

                # get tcn feature slices
                i3d_feature_jumping, i3d_feature_dropping, i3d_feature_entering, i3d_feature_ending = \
                    get_stage_features(i3d_feature, tcn_predict_time_points, predicted_points_num)

                # compute four stage output
                jumping = full_connect1(feature_link(i3d_feature_jumping))
                dropping = full_connect1(feature_link(i3d_feature_dropping))
                entering = full_connect1(feature_link(i3d_feature_entering))
                ending = full_connect1(feature_link(i3d_feature_ending))

                # predict final score
                predict_execute_score = full_connect2(torch.cat((jumping, dropping, entering, ending)))
                predict_final_score = (30-predict_execute_score) * difficulty_level

                # compute loss
                loss = compute_loss_score(predict_execute_score.float(), label_execute_score.float())
                epoch_loss += compute_loss_score(predict_final_score, label_final_score).item()

                # backward
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                truth_score.append(label_final_score.item())
                predict_score.append(predict_final_score.item())

            print("{:s}".format(split))
            print("epoch：{:03d}".format(epoch) + ", loss is {:.4f}".format(epoch_loss/len(dataloaders[split])))
            src, _ = stats.spearmanr(truth_score, predict_score)
            print("spearman coefficient is:{:.4f}".format(src))

            if src > best_src and split == 'test':
                best_src = src
                epoch_best = epoch
                print(
                    "find new full connect saving epoch {:3d}".format(epoch_best) + ", best coefficient:{:.4f}".format(best_src))
                if args.save:
                    torch.save({'epoch': epoch_best,
                                'full_connect1_model': full_connect1.state_dict(),
                                'full_connect2_model': full_connect2.state_dict(),
                                'feature_link': feature_link.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_coefficient': best_src}, f'ckpts/base_full_connect1.pt')


if __name__ == '__main__':

    args = get_parser().parse_args()

    init_seed(args)

    dataloaders = get_dataloaders(args)

    mstcn_model, feature_link, full_connect1, full_connect2, best_spearman_c = get_models(args)

    train_models(dataloaders, mstcn_model, feature_link, full_connect1, full_connect2, best_spearman_c, args)


