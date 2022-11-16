from __future__ import print_function

import argparse

__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser(description='CFLOW-AD')
    parser.add_argument('--dataset-path', default=r'D:\Datasets\mvtec', type=str, metavar='A',
                        help='where is the dataset tar file')
    parser.add_argument('--dataset', default='DAGM', type=str, metavar='D',
                        help='dataset name: mvtec (default: mvtec)')
    parser.add_argument('-cl', '--class-name', default='transistor', type=str, metavar='C',
                        help='class name for MVTec/STC (default: none)')
    parser.add_argument('-run', '--run-name', default=0, type=int, metavar='C',
                        help='name of the run (default: 0)')
    parser.add_argument('-inp', '--input-size', default=256, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument("--action-type", default='norm-train', type=str, metavar='T',
                        help='norm-train/norm-test (default: norm-train)')
    parser.add_argument('-bs', '--batch-size', default=2, type=int, metavar='B',
                        help='train batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-wd', '--weight-decay', type=float, default=3e-5)
    parser.add_argument('--sched-param', type=float, nargs='*', default=[0.995])
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of  epochs to train (default: 200')
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='saves test data visualizations')
    parser.add_argument('--workers', default=0, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default='0', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument(
        '--nominal-label', type=int, default=0,
        help='Determines the label that marks nominal samples. '  # 确定标记标称样本的类型
             'Note that this is not the class that is considered nominal! '  # 注意，这不是被认为是标称类的类!
             'For instance, class 5 is the nominal class, which is labeled with the nominal label 0.'
        # 例如，类5是标称类，它的标称标签为0
    )

    args = parser.parse_args()

    return args
