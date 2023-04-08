import csv
import math
import os
import pathlib
import random
import shutil
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from loguru import logger
from torchvision import transforms

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


def gen_img_fg(img):
    """
    生成图像前景蒙版
    """
    img_org = cv2.cvtColor(np.array(img.permute([1, 2, 0])), cv2.COLOR_RGB2GRAY)
    w, h = img_org.shape

    def no_shade(shade_img, mask):
        anti_mask = abs(mask - 1)
        shade_img = np.array(shade_img).astype(float)
        gaussian_img = cv2.GaussianBlur(shade_img, [21, 21], 0)
        no_shade_img = abs(cv2.log(shade_img.clip(1, 255)) - cv2.log(gaussian_img.clip(1, 255)))
        no_shade_img = 255 * ((no_shade_img - no_shade_img.min()) / (no_shade_img.max() - no_shade_img.min()))
        no_shade_img = (no_shade_img + np.sum(shade_img * anti_mask) / (np.sum(anti_mask) + 0.1)) * anti_mask + shade_img * mask
        no_shade_img = 255 * ((no_shade_img - no_shade_img.min()) / (no_shade_img.max() - no_shade_img.min()))
        return no_shade_img

    def corner(img_org, k, ratio):
        #     img_fg = abs(img[:15, :15].mean() - img)/4 + abs(img[-15:, :15].mean() - img)/4 + abs(img[:15, -15:].mean() - img)/4 + abs(img[-15:, -15:].mean() - img)/4
        img_fg = abs(img_org[:k, :k].mean() - img_org)
        avg_box = cv2.boxFilter(img_fg, -1, [int(w / 10), int(h / 10)])
        img_fg = (avg_box - avg_box.min()) / (avg_box.max() - avg_box.min())
        _, img_fg = cv2.threshold(img_fg, img_fg.max() * ratio, 1, cv2.THRESH_BINARY)
        img_fg = cv2.morphologyEx(img_fg, cv2.MORPH_OPEN, kernel=np.ones([5, 5]))
        img_fg = cv2.dilate(img_fg, kernel=np.ones([20, 20]))
        return img_fg

    img_org = no_shade(img_org, corner(img_org, 15, 0.25))
    img_fg = corner(img_org, 15, 0.4)
    # 如果检测出的前景区域较小，则认为整张图片都是前景
    if np.sum(img_fg) / np.prod(img_fg.shape) < 1 / 49:
        img_fg = np.ones_like(img_fg)

    return img_fg


def split_image(image, n):
    """将一张图像分成n * n个块"""
    width, height = image.size
    block_size = width // n
    block_list = []
    for i in range(n):
        for j in range(n):
            box = (j * block_size, i * block_size, (j + 1) * block_size, (i + 1) * block_size)
            block = image.crop(box)
            block_list.append(block)
    return block_list


def shuffle_blocks(block_list, n):
    """随机打乱块的位置，保证离中心越近的块随机后的位置也越靠近中心"""
    center_x = n // 2
    center_y = n // 2
    distance_dict = {}
    for i, block in enumerate(block_list):
        x = i % n
        y = i // n
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        distance_dict[i] = distance
    sorted_dict = sorted(distance_dict.items(), key=lambda x: x[1])
    new_block_list = []
    for item in sorted_dict:
        new_block_list.append(block_list[item[0]])
    return new_block_list


def generate_new_image(block_list, n):
    """生成新的图像"""
    width, height = block_list[0].size
    new_image = Image.new('RGB', (width * n, height * n))
    for i in range(n):
        for j in range(n):
            new_image.paste(block_list[i * n + j], (j * width, i * height))
    return new_image


def gen_shuffle_img(original_image, block_num=16, texture_radio=0.3):
    """
    将一张图像分割成 block_num * block_num 块，将这些块按照其到图像中心的距离进行排序
    取前 texture_radio 的块随机打乱顺序后生成一张新的图像
    然后将其平铺为原始图像大小
    block_num 必须能被 original_image 的宽高整除
    texture_radio 表示前景物体在所有块中所占比例
    """

    if isinstance(original_image, torch.Tensor):
        original_image = Image.fromarray(original_image.permute([1, 2, 0]).numpy())

    block_list = split_image(original_image, block_num)
    block_list = shuffle_blocks(block_list, block_num)
    block_total = len(block_list)

    block_list = block_list[:int(block_total * texture_radio)]

    block_list = block_list * (block_total // len(block_list)) + block_list[:block_total % len(block_list)]

    random.shuffle(block_list)

    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    block_list = list(map(lambda block: color_jitter(block), block_list))

    # 生成新的图像
    new_image = generate_new_image(block_list, block_num)
    new_image = np.array(new_image)

    return new_image


def interpolate(t):
    """
    让噪声像素间过渡更自然
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolate=interpolate
):
    """
    生成柏林噪声并二值化
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]] \
               .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolate(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def visualize_feature_map(tensor):
    """
    tensor shape should be (n, w, h) where n denote the number of grey feature.
    """
    tensor = tensor.unsqueeze(1).cpu().detach()

    x_min = tensor.view(tensor.shape[0], tensor.shape[1], -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
    x_max = tensor.view(tensor.shape[0], tensor.shape[1], -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)

    tensor = (tensor - x_min) / (x_max - x_min)

    grid = torchvision.utils.make_grid(tensor, nrow=int(math.sqrt(tensor.shape[0])), padding=8)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)
    return Image.fromarray(ndarr)


class DatasetConfig(ABC):
    home_dir_linux = r'/mnt/home/y21301045/datasets'

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
        return self.home_dir_linux + r"/mvtec"

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
        classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
        return list(reversed(classes))

    @property
    def batch_size(self):
        if self._debug:
            return 2
        return 42

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
        return self.home_dir_linux + r"/btad"

    @property
    def dataset_file_name(self):
        return "btad.zip"

    @property
    def dataset_extract_dir_name(self):
        return "extracted_btad_dataset"

    @property
    def dataset_dir_name(self):
        if self._debug:
            return "./BTech_Dataset_transformed"
        return './BTech_Dataset_transformed'

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
        return 42

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


class MPDD(DatasetConfig):
    def __init__(self, debug=False):
        self._debug = debug

    @property
    def dataset_path(self):
        if self._debug:
            return r'D:\Datasets\mpdd'
        return self.home_dir_linux + r"/mpdd"

    @property
    def dataset_file_name(self):
        return "mpdd.zip"

    @property
    def dataset_extract_dir_name(self):
        return "extracted_mpdd_dataset"

    @property
    def dataset_dir_name(self):
        return './'

    @property
    def normal_dir_label(self):
        return "good"

    @property
    def classes(self):
        return ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

    @property
    def batch_size(self):
        if self._debug:
            return 2
        return 42

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


class VISA(DatasetConfig):
    def __init__(self, debug=False):
        self._debug = debug

    @property
    def dataset_path(self):
        if self._debug:
            return r'D:\Datasets\visa'
        return self.home_dir_linux + r"/visa"

    @property
    def dataset_file_name(self):
        return "VisA_20220922.tar"

    @property
    def dataset_extract_dir_name(self):
        return "extracted_visa_dataset"

    @property
    def dataset_dir_name(self):
        format_data_dir = os.path.join(self.dataset_path, self.dataset_extract_dir_name, 'VisA_formatted')

        if not os.path.exists(format_data_dir):
            logger.info(f"format VsiA dataset: {format_data_dir}")
            extract_dir = os.path.join(self.dataset_path, self.dataset_extract_dir_name)
            image_meta_info_file = os.path.join(extract_dir, 'split_csv', '1cls.csv')

            with open(image_meta_info_file) as f:
                for img_item in csv.DictReader(f):
                    class_name, split, label, img_path, mask_path = img_item.values()
                    img_path = os.path.join(extract_dir, img_path)
                    img_name = os.path.basename(img_path)
                    if os.path.exists(img_path):
                        target_img_path = os.path.join(format_data_dir, class_name, split, self.normal_dir_label if label == 'normal' else 'defect')
                        os.makedirs(target_img_path, exist_ok=True)
                        shutil.copyfile(img_path, os.path.join(target_img_path, img_name))
                        mask_path = os.path.join(extract_dir, mask_path)
                        if os.path.exists(mask_path) and os.path.isfile(mask_path):
                            target_mask_path = os.path.join(format_data_dir, class_name, 'ground_truth', 'defect')
                            os.makedirs(target_mask_path, exist_ok=True)
                            shutil.copyfile(os.path.join(extract_dir, mask_path), os.path.join(target_mask_path, img_name))

        return os.path.basename(format_data_dir)

    @property
    def normal_dir_label(self):
        return "good"

    @property
    def classes(self):
        return ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    @property
    def batch_size(self):
        if self._debug:
            return 2
        return 42

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
