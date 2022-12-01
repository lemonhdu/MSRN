import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save',
                        action='store_true',
                        help='if set true, save the best model',
                        default=True)

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=5e-5)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-5)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=1024)

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=8)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=1)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=1)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=100)

    return parser


