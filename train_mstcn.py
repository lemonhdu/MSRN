import os

from torch import optim

from config import get_parser
from models.mstcn import MultiStageModel

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from params import *
from dataset import VideoDataset
from utils import *


def get_dataloaders(args):
    '''
    get train set and test set for tcn. The number of videos (.npy) in train set is 300.
    '''

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn,
                                                       )

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn
                                                      )
    return dataloaders


def get_models(args):
    '''
    set your configs in config.py
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    mstcn_model = MultiStageModel(num_stages=tcn_num_stages,
                            num_layers=tcn_num_layers,
                            num_f_maps=tcn_num_f_maps,
                            dim=tcn_features_dim,
                            num_classes=tcn_num_classes).cuda()

    return mstcn_model


def compute_loss_tcn(predictions, tcn_labels, num_classes, mask):
    '''
    compute loss for tcn-network
    '''
    loss = 0
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_mse = nn.MSELoss(reduction='none')
    for p in predictions:
        loss += criterion_ce(p.transpose(2, 1).contiguous().view(-1, num_classes), tcn_labels.view(-1))
        loss += 0.15 * torch.mean(torch.clamp(criterion_mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                              min=0, max=16) * mask[:, :, 1:])
    return loss


def train_models(dataloaders, mstcn_model, args):

    optimizer = optim.Adam(mstcn_model.parameters(), lr=args.lr)

    tcn_best_acc = 0
    for epoch in range(args.num_epochs):

        for split in ['train', 'test']:

            if split == 'train':
                mstcn_model.train()
                torch.set_grad_enabled(True)
            else:
                mstcn_model.eval()
                torch.set_grad_enabled(False)

            epoch_loss = 0
            correct = 0
            total = 0

            for data in tqdm(dataloaders[split]):

                i3d_features = data['video'].cuda()

                tcn_labels = data['tcn_time_labels']

                # ensure that the length of labels for training is equal to that of i3d_features
                tcn_labels = (tcn_labels.squeeze(0))[:, :len(i3d_features[0][0])].cuda()

                mask = torch.ones(1, tcn_num_classes, len(tcn_labels[0]), dtype=torch.float).cuda()

                predictions = mstcn_model(i3d_features, mask)

                loss_tcn = compute_loss_tcn(predictions, tcn_labels, tcn_num_classes, mask)

                epoch_loss += loss_tcn.item()

                if split == 'train':
                    optimizer.zero_grad()
                    loss_tcn.backward()
                    optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == tcn_labels).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            tcn_acc = correct / total
            print("{:s}".format(split) + ", epoch：{:03d}".format(epoch) + ", loss is {:03f}".format(epoch_loss/len(dataloaders[split])) +
                  ", correct={:03f}".format(correct/total))

        if tcn_acc > tcn_best_acc:
            tcn_best_acc = tcn_acc
            epoch_best = epoch
            print("find new tcn saving epoch {:3d}".format(epoch_best)+", best accuracy:{:.4f}".format(tcn_best_acc))
            if args.save:
                torch.save({'epoch': epoch_best,
                            'tcn_model': mstcn_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'tcn_best_acc': tcn_best_acc}, f'ckpts/mstcn02.pt')


if __name__ == '__main__':

    args = get_parser().parse_args()

    dataloaders = get_dataloaders(args)

    mstcn_model = get_models(args)

    train_models(dataloaders, mstcn_model, args)
