from abc import ABC, abstractmethod
from typing import Tuple
from collections import Counter

import numpy as np
import torch
from datasets.noise_modes import generate_noise
from datasets.offline_supervisor import noise as apply_noise, malformed_normal as apply_malformed_normal
from datasets.preprocessing import get_target_label_idx
from logging import Logger
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
from config import TrainConfigures


class BaseADDataset(ABC):
    """ Anomaly detection dataset base class """

    def __init__(self, root: str, logger: Logger = None):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self._train_set = None  # must be of type torch.utils.data.Dataset
        self._test_set = None  # must be of type torch.utils.data.Dataset

        self.shape = None  # shape of datapoints, c x h x w
        self.raw_shape = None  # shape of datapoint before preprocessing is applied, c x h x w

        self.logger = logger

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> Tuple[
        DataLoader, DataLoader]:
        """ Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set. """
        pass

    def __repr__(self):
        return self.__class__.__name__

    def logprint(self, s: str, fps: bool = False):
        """ prints a string via the logger """
        if self.logger is not None:
            self.logger.print(s, fps)
        else:
            print(s)


class TorchvisionDataset(BaseADDataset):
    """ TorchvisionDataset class for datasets already implemented in torchvision.datasets
    TorchvisionDataset类用于已经在torchvision.dataset中实现的数据集
    """

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    def __init__(self, root: str, logger=None):
        super().__init__(root, logger=logger)

    def loaders(self, batch_size: int = 2, shuffle_train=True, shuffle_test=False, num_workers: int = 0, train: bool = True) \
            -> Tuple[DataLoader, DataLoader]:
        assert not shuffle_test, \
            'using shuffled test raises problems with original GT maps for GT datasets, thus disabled atm!'  # 使用混合测试会导致GT数据集的原始GT地图出现问题，因此禁用了atm!
        # classes = None means all classe
        if train:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=False,
                                      prefetch_factor=2, persistent_workers=TrainConfigures.dataset.worker_num > 0)
            test_loader = DataLoader(dataset=self.test_set, batch_size=5, shuffle=False,
                                     num_workers=num_workers, pin_memory=True, drop_last=False,
                                     prefetch_factor=2, persistent_workers=TrainConfigures.dataset.worker_num > 0)

            return train_loader, test_loader
        else:
            test_loader = DataLoader(dataset=self.test_set, batch_size=5, shuffle=False,
                                     num_workers=num_workers, pin_memory=True, drop_last=True)

            return test_loader

    def preview(self, percls=20, train=True) -> torch.Tensor:
        """
        Generates a preview of the dataset, i.e. it generates an image of some randomly chosen outputs
        of the dataloader, including ground-truth maps if available.
        生成数据集的预览，即生成一些随机选择的输出的图像包括地面实况地图(如果有的话)。
        The data samples already have been augmented by the preprocessing pipeline.
        数据样本已经通过预处理管道进行了扩充
        This method is useful to have an overview of how the preprocessed samples look like and especially
        to have an early look at the artificial anomalies.
        这种方法有助于对预处理样品的外观有一个概述，特别是先看看这些人为异常。
        :param percls: how many samples are shown per class, i.e. for anomalies and nominal samples each
        每类显示多少样本，即异常样本和标称样本
        :param train: whether to show training samples or test samples
        是显示训练样本还是测试样本
        :return: a Tensor of images (n x c x h x w)
        """
        self.logprint('Generating dataset preview...')
        # assert num_workers>0, otherwise the OnlineSupervisor is initialized with the same shuffling in later workers
        if train:
            loader, _ = self.loaders(10, num_workers=1, shuffle_train=True)
        else:
            _, loader = self.loaders(10, num_workers=1, shuffle_test=True)
        x, y, gts, out = torch.FloatTensor(), torch.LongTensor(), torch.FloatTensor(), []  # 返回空tensor
        if isinstance(self.train_set, GTMapADDataset):
            for xb, yb, gtsb in loader:
                x, y, gts = torch.cat([x, xb]), torch.cat([y, yb]), torch.cat([gts, gtsb])
                if all([x[y == c].size(0) >= percls for c in [0, 1]]):
                    break
        else:
            for xb, yb in loader:
                x, y = torch.cat([x, xb]), torch.cat([y, yb])
                if all([x[y == c].size(0) >= percls for c in [0, 1]]):
                    break
        for c in sorted(set(y.tolist())):
            out.append(x[y == c][:percls])  # 取20个样本
        if len(gts) > 0:
            assert len(set(gts.reshape(-1).tolist())) <= 2, 'training process assumes zero-one gtmaps'  # 训练过程假定gtmaps为0 - 1
            out.append(torch.zeros_like(x[:percls]))  # 生成20个全零形状为x的张量
            for c in sorted(set(y.tolist())):
                g = gts[y == c][:percls]
                if x.shape[1] > 1:
                    g = g.repeat(1, x.shape[1], 1, 1)
                out.append(g)
        self.logprint('Dataset preview generated.')
        return torch.stack([o[:min(Counter(y.tolist()).values())] for o in out])

    def _generate_artificial_anomalies_train_set(self, supervise_mode: str, noise_mode: str, oe_limit: int,
                                                 train_set: Dataset, nom_class: int):
        """
        This method generates offline artificial anomalies,
        i.e. it generates them once at the start of the training and adds them to the training set.
        It creates a balanced dataset, thus sampling as many anomalies as there are nominal samples.
        This is way faster than online generation, but lacks diversity (hence usually weaker performance).
        这种方法产生离线人工异常，即，它在训练开始时生成它们一次，并将它们添加到训练集中。
        它创建了一个平衡的数据集，因此可以对标称样本一样多的异常进行采样
        :param supervise_mode: the type of generated artificial anomalies.
            unsupervised: no anomalies, returns a subset of the original dataset containing only nominal samples.
            无异常，返回只包含标称样本的原始数据集的子集。
            other: other classes, i.e. all the true anomalies! 其他类，即所有真正的异常!
            noise: pure noise images (can also be outlier exposure based). 纯噪声图像(也可以是基于离群曝光)。
            malformed_normal: add noise to nominal samples to create malformed nominal anomalies. 向标称样本添加噪声以产生畸形的标称异常。
            malformed_normal_gt: like malformed_normal, but also creates artificial ground-truth maps
            像malformed_normal，但也创建了人工地面真实地图，在原始标称样本之间的差异标记像素异常，而畸形型则大于低阈值。
                that mark pixels anomalous where the difference between the original nominal sample
                and the malformed one is greater than a low threshold.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: the number of different outlier exposure samples used in case of outlier exposure based noise.
        在基于噪声的离群值暴露中使用的不同离群值暴露样本的数量。
        :param train_set: the training set that is to be extended with artificial anomalies.  表示数据集的抽象类
        :param nom_class: the class considered nominal
        :return:
        """
        if isinstance(train_set.targets, torch.Tensor):
            dataset_targets = train_set.targets.clone().data.cpu().numpy()  # 将tensor转换成numpy
        else:  # e.g. imagenet
            dataset_targets = np.asarray(train_set.targets)
        train_idx_normal = get_target_label_idx(dataset_targets, self.normal_classes)  # 获取正常样本的索引的列表
        generated_noise = norm = None
        if supervise_mode not in ['unsupervised', 'other']:
            self.logprint('Generating artificial anomalies...')  # 打印生成人工异常
            generated_noise = self._generate_noise(
                noise_mode, train_set.data[train_idx_normal].shape, oe_limit,
                self.root
            )
            norm = train_set.data[train_idx_normal]
        if supervise_mode in ['other']:
            self._train_set = train_set
        elif supervise_mode in ['unsupervised']:  # 如果监督模式为无监督，即训练样本都为正常样本
            if isinstance(train_set, GTMapADDataset):
                self._train_set = GTSubset(train_set, train_idx_normal)  # _train_set 包括 data,label, gtmap, GTSubset类实例化
            else:
                self._train_set = Subset(train_set, train_idx_normal)  # 返回正常的样本  Subset类实例化
        elif supervise_mode in ['noise']:
            self._train_set = apply_noise(self.outlier_classes, generated_noise, norm, nom_class, train_set)
        elif supervise_mode in ['malformed_normal']:
            self._train_set = apply_malformed_normal(self.outlier_classes, generated_noise, norm, nom_class, train_set)
        elif supervise_mode in ['malformed_normal_gt']:
            train_set, gtmaps = apply_malformed_normal(
                self.outlier_classes, generated_noise, norm, nom_class, train_set, gt=True
            )
            self._train_set = GTMapADDatasetExtension(train_set, gtmaps)
        else:
            raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
        if supervise_mode not in ['unsupervised', 'other']:
            self.logprint('Artificial anomalies generated.')

    def _generate_noise(self, noise_mode: str, size: torch.Size, oe_limit: int = None, datadir: str = None, img_fg=None) -> torch.Tensor:
        generated_noise = generate_noise(noise_mode, size, oe_limit, logger=self.logger, datadir=datadir, img_fg=img_fg)
        return generated_noise


class ThreeReturnsDataset(Dataset):
    """ Dataset base class returning a tuple of three items as data samples """

    @abstractmethod
    def __getitem__(self, index):
        return None, None, None


class GTMapADDataset(ThreeReturnsDataset):
    """ Dataset base class returning a tuple (input, label, ground-truth map) as data samples
    返回元组(input, label, ground-truth map)作为数据样本的Dataset基类
    """

    @abstractmethod
    def __getitem__(self, index):
        x, y, gtmap = None, None, None
        return x, y, gtmap


class GTSubset(Subset, GTMapADDataset):
    """ Subset base class for GTMapADDatasets """
    pass


class GTMapADDatasetExtension(GTMapADDataset):
    """
    This class is used to extend a regular torch dataset such that is returns the corresponding ground-truth map
    in addition to the usual (input, label) tuple.
    """

    def __init__(self, dataset: Dataset, gtmaps: torch.Tensor, overwrite=True):
        """
        :param dataset: a regular torch dataset
        :param gtmaps: a tensor of ground-truth maps (n x h x w)
        :param overwrite: if dataset is already a GTMapADDataset itself,
            determines if gtmaps of dataset shall be overwritten.
            None values of found gtmaps in dataset are overwritten in any case.
        """
        self.ds = dataset
        self.extended_gtmaps = gtmaps
        self.overwrite = overwrite
        if isinstance(self.ds, GTMapADDataset):
            assert hasattr(self.ds, 'gt')
            if self.ds.gt is None:
                self.ds.gt = gtmaps.mul(255).byte()
                self.overwrite = False

    @property
    def targets(self):
        return self.ds.targets

    @property
    def data(self):
        return self.ds.data

    def __getitem__(self, index: int):
        gtmap = self.extended_gtmaps[index]

        if isinstance(self.ds, GTMapADDataset):
            x, y, gt = self.ds[index]
            if self.overwrite or gt is None:
                gt = gtmap
            res = (x, y, gt)
        else:
            res = (*self.ds[index], gtmap)

        return res

    def __len__(self):
        return len(self.ds)
