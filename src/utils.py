import os
import pathlib
from abc import ABC, abstractmethod

import torch
from loguru import logger

RESULT_DIR = './results'
WEIGHT_DIR = './weights'
MODEL_DIR = './models'

__all__ = ('save_results', 'save_weights', 'load_weights', 'warmup_learning_rate')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, model_name, class_name, run_date):
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    fp = open(os.path.join(RESULT_DIR, '{}_{}.txt'.format(model_name, run_date)), "w")
    fp.write(result)
    fp.close()


def save_weights(encoder, decoders, model_name, run_date):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': [decoder.state_dict() for decoder in decoders]}
    filename = '{}_{}.pt'.format(model_name, run_date)
    path = os.path.join(WEIGHT_DIR, filename)
    torch.save(state, path)
    logger.info('Saving weights to {}'.format(filename))


def load_weights(encoder, decoders, filename):
    path = os.path.join(filename)
    state = torch.load(path)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    logger.info('Loading weights from {}'.format(filename))


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate


def get_dir(*folders):
    folder = os.path.join(*folders)
    if not os.path.exists(folder):
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"create dir: {folder}")
    return folder


class DatasetConfig(ABC):

    @property
    def dataset_path(self):
        """
        数据集位置（mvtec_anomaly_detection.tar.xz 存放位置）
        """
        return

    @property
    def dataset_file_name(self):
        return

    @property
    def dataset_extract_dir_name(self):
        """
        注1：该目录在 dataset_path 目录下
        注2：如果解压文件夹已经存在，则不会重新解压
        """
        return

    @property
    def dataset_dir_name(self):
        """
        该目录在 dataset_extract_dir_name 目录下
        该目录下级目录需为classes对应的名称
        """
        return

    @property
    def normal_dir_label(self):
        """
        不同数据集中，存放正常样本的文件夹名称不同
        例如在 MVTec 中使用的是 good，而在 BTAD 中使用的是 ok
        """
        return

    @property
    def classes(self):
        """
        待训练的类别
        """
        return

    @property
    def batch_size(self):
        """
        单次训练批次数量
        """
        return

    @property
    def worker_num(self):
        """
        pytorch数据集预加载线程数
        """
        return

    @abstractmethod
    def find_ground_truth(self, img_path):
        return None


class MVTec(DatasetConfig):
    def __init__(self, debug=False):
        self._debug = debug

    @property
    def dataset_path(self):
        if self._debug:
            return r'D:\Datasets\mvtec'
        return r"/mnt/home/y21301045/datasets/mvtec"

    @property
    def dataset_file_name(self):
        return "mvtec_anomaly_detection.tar.xz"

    @property
    def dataset_extract_dir_name(self):
        return "extracted_mvtec_dataset"

    @property
    def dataset_dir_name(self):
        return "./"

    @property
    def normal_dir_label(self):
        return "good"

    @property
    def classes(self):
        return ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
        # return ['toothbrush', 'capsule', 'screw', 'pill', 'carpet', 'cable', 'transistor', 'metal_nut', 'tile', 'wood', 'bottle', 'hazelnut', 'leather', 'grid', 'zipper']

    @property
    def batch_size(self):
        if self._debug:
            return 2
        return 46

    @property
    def worker_num(self):
        if self._debug:
            return 0
        return 12

    def find_ground_truth(self, img_path):
        img_path_split = os.path.split(img_path)
        anorm_class_name = os.path.split(img_path_split[0])[1]
        anorm_gt_dir = os.path.abspath(os.path.join(img_path, '../../../ground_truth', anorm_class_name))
        for gt_name in os.listdir(anorm_gt_dir):
            if str(gt_name).split('.')[0] == img_path_split[1].split('.')[0] + "_mask":
                return os.path.join(anorm_gt_dir, gt_name)
        else:
            logger.error(f"No match ground-truth file found in {anorm_gt_dir} for {img_path}")
            return None


class BTAD(DatasetConfig):
    def __init__(self, debug=False):
        self._debug = debug

    @property
    def dataset_path(self):
        if self._debug:
            return r'D:\Datasets\btad'
        return r"/mnt/home/y21301045/datasets/btad"

    @property
    def dataset_file_name(self):
        return "btad.zip"

    @property
    def dataset_extract_dir_name(self):
        return "extracted_btad_dataset"

    @property
    def dataset_dir_name(self):
        return "BTech_Dataset_transformed"

    @property
    def normal_dir_label(self):
        return "ok"

    @property
    def classes(self):
        return ['01', '02', '03']

    @property
    def batch_size(self):
        if self._debug:
            return 2
        return 46

    @property
    def worker_num(self):
        if self._debug:
            return 0
        return 12

    def find_ground_truth(self, img_path):
        img_path_split = os.path.split(img_path)
        anorm_class_name = os.path.split(img_path_split[0])[1]
        anorm_gt_dir = os.path.abspath(os.path.join(img_path, '../../../ground_truth', anorm_class_name))
        for gt_name in os.listdir(anorm_gt_dir):
            if str(gt_name).split('.')[0] == img_path_split[1].split('.')[0]:
                return os.path.join(anorm_gt_dir, gt_name)
        else:
            logger.error(f"No match ground-truth file found in {anorm_gt_dir} for {img_path}")
            return None
