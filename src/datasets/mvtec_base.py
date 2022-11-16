import os
import tarfile
from logging import Logger
from typing import Callable
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.imagenet import check_integrity, verify_str_arg

from datasets.bases import GTMapADDataset


# from torchvision.datasets.utils import _is_gzip, _is_tar, _is_targz, _is_zip


class MvTec(VisionDataset, GTMapADDataset):
    """ Implemention of a torch style MVTec dataset """
    url = "ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz"
    dataset_file_name = "mvtec_anomaly_detection.tar.xz"
    base_folder = 'mvtec'
    # base_folder = 'BTAD'
    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )
    # labels = ('01', '02', '03')
    normal_anomaly_label = 'good'
    normal_anomaly_label_idx = 0

    def __init__(self, root: str, split: str = 'train', target_transform: Callable = None,
                 img_gt_transform: Callable = None, transform: Callable = None, all_transform: Callable = None,
                 shape=(3, 300, 300), normal_classes=(), nominal_label=0, anomalous_label=1,
                 logger: Logger = None, enlarge: bool = False, flag: bool = False,
                 ):
        """
        Loads all data from the prepared torch tensors. If such torch tensors containg MVTec data are not found
        in the given root directory, instead downloads the raw data and prepares the tensors.
        从准备好的torch张量加载所有数据。 如果在给定的根目录中找不到包含 MVTec 数据的此类 Torch 张量，则下载原始数据并准备张量。
        They contain labels, images, and ground-truth maps for a fixed size, determined by the shape parameter.
        它们包含由shape参数确定的固定大小的标签、图像和地面实况图。
        :param root: directory where the data is to be found.
        :param split: whether to use "train", "test", or "test_anomaly_label_target" data.
            In the latter case the get_item method returns labels indexing the anomalous class rather than
            the object class. That is, instead of returning 0 for "bottle", it returns "1" for "large_broken".
        :param target_transform: function that takes label and transforms it somewhat.接受标签并对其进行一些转换的函数。
            Target transform is the first transform that is applied.目标变换是应用的第一个变换。
        :param img_gt_transform: function that takes image and ground-truth map and transforms it somewhat.
        获取图像和地面实况图并对其进行一些转换的函数。
            Useful to apply the same augmentation to image and ground-truth map (e.g. cropping), s.t.
            用于将相同的增强应用于图像和地面实况图（例如裁剪），s.t.
            the ground-truth map still matches the image.
            真实地图仍然与图像匹配
            ImgGt transform is the third transform that is applied.
            ImgGt 变换是应用的第三个变换。
        :param transform: function that takes image and transforms it somewhat.获取图像并对其进行一些转换的函数。
            Transform is the last transform that is applied.变换是最后应用的变换。
        :param all_transform: function that takes image, label, and ground-truth map and transforms it somewhat.
        获取图像、标签和地面实况图并对其进行一些转换的函数。
            All transform is the second transform that is applied.所有变换是应用的第二个变换。
        :param download: whether to download if data is not found in root.
        :param shape: the shape (c x h x w) the data should be resized to (images and ground-truth maps).
        数据的形状（c x h x w）应调整为（图像和地面实况图）。
        :param normal_classes: all the classes that are considered nominal (usually just one).
        :param nominal_label: the label that is to be returned to mark nominal samples. 要返回以标记标称样品的标签。
        :param anomalous_label: the label that is to be returned to mark anomalous samples.
        :param logger: logger
        :param enlarge: whether to enlarge the dataset, i.e. repeat all data samples ten times.
            Consequently, one iteration (epoch) of the data loader returns ten times as many samples.
            This speeds up loading because the MVTec-AD dataset has a poor number of samples and
            PyTorch requires additional work in between epochs.
        """
        super(MvTec, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "test", "test_anomaly_label_target"))
        self.img_gt_transform = img_gt_transform
        self.all_transform = all_transform
        self.shape = shape
        self.orig_gtmaps = None
        self.normal_classes = normal_classes
        self.nominal_label = nominal_label
        self.anom_label = anomalous_label
        self.logger = logger
        self.enlarge = enlarge
        self.flag = flag

        self.process_data(shape=self.shape[1:])

        print(f"load dataset from {self.data_file}")
        dataset_dict = torch.load(self.data_file)
        self.anomaly_label_strings = dataset_dict['anomaly_label_strings']
        if self.split == 'train':
            self.data, self.targets = dataset_dict['train_data'], dataset_dict['train_labels']
            self.gt, self.anomaly_labels = None, None
        else:
            self.data, self.targets = dataset_dict['test_data'], dataset_dict['test_labels']
            self.gt, self.anomaly_labels = dataset_dict['test_maps'], dataset_dict['test_anomaly_labels']

        if self.enlarge:
            self.data, self.targets = self.data.repeat(10, 1, 1, 1), self.targets.repeat(10)
            self.gt = self.gt.repeat(10, 1, 1) if self.gt is not None else None
            self.anomaly_labels = self.anomaly_labels.repeat(10) if self.anomaly_labels is not None else None
            self.orig_gtmaps = self.orig_gtmaps.repeat(10, 1, 1) if self.orig_gtmaps is not None else None

        if self.nominal_label != 0:
            print('Swapping labels, i.e. anomalies are 0 and nominals are 1, same for GT maps.')
            assert -3 not in [self.nominal_label, self.anom_label]
        print('Dataset complete.')

        self.center_ = transforms.Compose([transforms.ToPILImage(),
                                           transforms.CenterCrop(260),
                                           transforms.ToTensor()])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img, label = self.data[index], self.targets[index]  # 都为张量 c×h×w  label为单个值

        if self.split == 'test_anomaly_label_target':
            label = self.target_transform(self.anomaly_labels[index])

        if self.target_transform is not None:
            label = self.target_transform(label)  # 将目标标签换为 0或1，target若非正常类标签，则label为1，否则为0

        if self.split == 'train' and self.gt is None:
            # img = self.center_(img)
            assert self.anom_label in [0, 1]
            # gt is assumed to be 1 for anoms always (regardless of the anom_label), since the supervisors work that way
            # 对于anoms, Gt总是假定为1(不管是否有anom_label)，因为管理器是这样工作的
            # later code fixes that (and thus would corrupt it if the correct anom_label is used here in swapped case)
            # 后面的代码修复了这个问题(如果在交换的情况下使用了正确的anom_label，则会损坏它)
            gtinitlbl = label if self.anom_label == 1 else (1 - label)
            gt = (torch.ones_like(img)[0] * gtinitlbl).mul(255).byte()  # 异常图像gt为255，正常图像gt为0
        else:
            gt = self.gt[index]

        if self.all_transform is not None:
            img, gt, label = self.all_transform((img, gt, label))
            gt = gt.mul(255).byte() if gt.dtype != torch.uint8 else gt
            img = img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else img
        # doing this so that it is consistent with all other datasets 这样做是为了与所有其他数据集一致
        # to return a PIL Image
        img = Image.fromarray(img.transpose(0, 2).transpose(0, 1).numpy(), mode='RGB')
        gt = Image.fromarray(gt.squeeze(0).numpy(), mode='L')

        if self.img_gt_transform is not None:
            # img = self.img_gt_transform(img)
            # gt = self.img_gt_transform(gt)
            img, gt = self.img_gt_transform((img, gt))

        if self.transform is not None:
            img = self.transform(img)

        if self.nominal_label != 0:
            gt[gt == 0] = -3  # -3 is chosen arbitrarily here
            gt[gt == 1] = self.anom_label
            gt[gt == -3] = self.nominal_label

        return img, label, gt

    def __len__(self) -> int:
        return len(self.data)

    def process_data(self, verbose=True, shape=None, cls=None):
        assert shape is not None or cls is not None, 'original shape requires a class'
        # 如果admvtec_224×224.pt文件存在，下面语句不执行
        if not check_integrity(self.data_file if shape is not None else self.orig_data_file(cls)):
            # 解压文件
            extract_dir = self.extract_archive(os.path.join(self.root, self.dataset_file_name))
            train_data, train_labels = [], []
            test_data, test_labels, test_maps, test_anomaly_labels = [], [], [], []
            anomaly_labels, albl_idmap = [], {self.normal_anomaly_label: self.normal_anomaly_label_idx}

            for lbl_idx, lbl in enumerate(self.labels if cls is None else [self.labels[cls]]):
                if verbose:
                    print('Processing data for label {}'.format(lbl))
                for anomaly_label in sorted(os.listdir(os.path.join(extract_dir, lbl, 'test'))):  # os.listdir 返回路径下文件名组成的列表
                    for img_name in sorted(os.listdir(os.path.join(extract_dir, lbl, 'test', anomaly_label))):  # 返回异常文件下的图片名
                        with open(os.path.join(extract_dir, lbl, 'test', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        if anomaly_label != self.normal_anomaly_label:  # 图像为异常类型
                            mask_name = self.convert_img_name_to_mask_name(img_name)
                            with open(os.path.join(extract_dir, lbl, 'ground_truth', anomaly_label, mask_name), 'rb') as f:
                                mask = Image.open(f)
                                mask = self.img_to_torch(mask, shape)
                        else:
                            mask = torch.zeros_like(sample)
                        test_data.append(sample)
                        test_labels.append(cls if cls is not None else lbl_idx)
                        test_maps.append(mask)
                        if anomaly_label not in albl_idmap:
                            albl_idmap[anomaly_label] = len(albl_idmap)
                        test_anomaly_labels.append(albl_idmap[anomaly_label])

                for anomaly_label in sorted(os.listdir(os.path.join(extract_dir, lbl, 'train'))):
                    for img_name in sorted(os.listdir(os.path.join(extract_dir, lbl, 'train', anomaly_label))):
                        with open(os.path.join(extract_dir, lbl, 'train', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        train_data.append(sample)
                        train_labels.append(lbl_idx)

            anomaly_labels = list(zip(*sorted(albl_idmap.items(), key=lambda kv: kv[1])))[0]
            train_data = torch.stack(train_data)
            train_labels = torch.IntTensor(train_labels)
            test_data = torch.stack(test_data)
            test_labels = torch.IntTensor(test_labels)
            test_maps = torch.stack(test_maps)[:, 0, :, :]  # r=g=b -> grayscale
            test_anomaly_labels = torch.IntTensor(test_anomaly_labels)
            '''
            train_labels: 类别数字化0-14,如0表示bottle,1表示cable等
            test_labels:类别数字化，如0表示bottle
            test_anomaly_labels:具体异常数字化，如0表示good
            anomaly_label_strings:如，good
            '''
            torch.save(
                {
                    'train_data': train_data,
                    'train_labels': train_labels,
                    'test_data': test_data,
                    'test_labels': test_labels,
                    'test_maps': test_maps,
                    'test_anomaly_labels': test_anomaly_labels,
                    'anomaly_label_strings': anomaly_labels
                },
                self.data_file if shape is not None else self.orig_data_file(cls)
            )

            # cleanup temp directory
            # for dirpath, dirnames, filenames in os.walk(extract_dir):
            #     os.chmod(dirpath, 0o755)
            #     for filename in filenames:
            #         os.chmod(os.path.join(dirpath, filename), 0o755)
            # shutil.rmtree(extract_dir)
        else:
            print(f'Dataset has been processed: {self.data_file if shape is not None else self.orig_data_file(cls)}')
            return

    def get_original_gtmaps_normal_class(self) -> torch.Tensor:
        """
        Returns ground-truth maps of original size for test samples.
        The class is chosen according to the normal class the dataset was created with.
        该类是根据创建数据集时使用的普通类来选择的。
        This method is usually used for pixel-wise ROC computation.
        """
        assert self.split != 'train', 'original maps are only available for test mode'
        assert len(self.normal_classes) == 1, 'normal classes must be known and there must be exactly one'
        assert self.all_transform is None, 'all_transform would be skipped here'
        assert all([isinstance(t, (transforms.Resize, transforms.ToTensor)) for t in self.img_gt_transform.transforms])
        if self.orig_gtmaps is None:
            self.process_data(shape=None, cls=self.normal_classes[0])
            orig_ds = torch.load(self.orig_data_file(self.normal_classes[0]))
            self.orig_gtmaps = orig_ds['test_maps'].unsqueeze(1).div(255)
        return self.orig_gtmaps

    @property
    def data_file(self):
        return os.path.join(self.root, self.filename)

    @property
    def filename(self):
        return "admvtec_{}x{}.pt".format(self.shape[1], self.shape[2])
        # return "BTAD_{}x{}.pt".format(self.shape[1], self.shape[2])

    def orig_data_file(self, cls):
        return os.path.join(self.root, self.orig_filename(cls))

    def orig_filename(self, cls):
        return "admvtec_orig_cls{}.pt".format(cls)
        # return "BTAD_orig_cls{}.pt".format(cls)

    @staticmethod
    def img_to_torch(img, shape=None):
        if shape is not None:
            return torch.nn.functional.interpolate(
                torch.from_numpy(np.array(img.convert('RGB'))).float().transpose(0, 2).transpose(1, 2)[None, :],
                shape
            )[0].byte()
        else:
            return torch.from_numpy(np.array(img.convert('RGB'))).float().transpose(0, 2).transpose(1, 2)[None, :][0].byte()

    @staticmethod
    def convert_img_name_to_mask_name(img_name):
        return img_name.replace('.png', '_mask.png')

    @staticmethod
    def extract_archive(dataset_tar_file: str) -> str:
        assert len(dataset_tar_file) > 0 and dataset_tar_file.endswith('.tar.xz'), 'invalid dataset source file'
        file_path, file_name = os.path.split(dataset_tar_file)
        extract_dir = os.path.join(file_path, 'extracted')

        with tarfile.open(dataset_tar_file, 'r:xz') as tar:
            print(f"extracting dataset tar file: {dataset_tar_file} to {extract_dir}")
            tar.extractall(path=extract_dir)

        return extract_dir
