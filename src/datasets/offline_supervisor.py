from typing import List

import torch
from torch.utils.data.dataset import Dataset


def noise(outlier_classes: List[int], generated_noise: torch.Tensor, norm: torch.Tensor,
          nom_class: int, train_set: Dataset, gt: bool = False) -> Dataset:
    """
    Creates a dataset based on the nominal classes of a given dataset and generated noise anomalies.
    :param outlier_classes: a list of all outlier class indices.
    :param generated_noise: torch tensor of noise images (might also be Outlier Exposure based noise) (n x c x h x w).
    :param norm: torch tensor of nominal images (n x c x h x w).
    :param nom_class: the index of the class that is considered nominal.
    :param train_set: some training dataset.
    :param gt: whether to provide ground-truth maps as well, atm not available!
    :return: a modified dataset, with training data consisting of nominal samples and artificial anomalies.
    """
    if gt:
        raise ValueError('No GT mode for pure noise available!')
    anom = generated_noise.clamp(0, 255).byte()  # 使张量的值在0到255之间
    data = torch.cat((norm, anom))
    targets = torch.cat(
        (torch.ones(norm.size(0)) * nom_class,
         torch.ones(anom.size(0)) * outlier_classes[0])
    )
    train_set.data = data
    train_set.targets = targets
    return train_set


def malformed_normal(outlier_classes: List[int], generated_noise: torch.Tensor, norm: torch.Tensor, nom_class: int,
                     train_set: Dataset, gt: bool = False, brightness_threshold: float = 0.11*255) -> Dataset:
    """
    Creates a dataset based on the nominal classes of a given dataset and generated noise anomalies.
    基于给定数据集的标称类创建数据集并生成噪声异常。
    Unlike above, the noise images are not directly utilized as anomalies, but added to nominal samples to
    create malformed normal anomalies.
    与上面不同的是，噪声图像不是直接作为异常来使用，而是添加到标称样本中来创建畸形的正常异常。
    :param outlier_classes: a list of all outlier class indices.所有离群值类索引的列表
    :param generated_noise: torch tensor of noise images (might also be Outlier Exposure based noise) (n x c x h x w).
    噪声图像的torch张量(也可能是基于离群曝光的噪声)
    :param norm: torch tensor of nominal images (n x c x h x w).
    :param nom_class: the index of the class that is considered nominal.
    :param train_set: some training dataset.
    :param gt: whether to provide ground-truth maps as well.
    :param brightness_threshold: if the average brightness (averaged over color channels) of a pixel exceeds this
        threshold, the noise image's pixel value is subtracted instead of added.
        如果一个像素的平均亮度(在颜色通道上的平均亮度)超过这个阈值，噪声图像的像素值将被减去而不是增加。
        This avoids adding brightness values to bright pixels, where approximately no effect is achieved at all.
        这避免了为明亮像素添加亮度值，在那里几乎没有实现任何效果。
    :return: a modified dataset, with training data consisting of nominal samples and artificial anomalies.
    修正数据集，训练数据由标称样本和人工异常组成。
    """
    assert (norm.dim() == 4 or norm.dim() == 3) and generated_noise.shape == norm.shape
    norm_dim = norm.dim()
    if norm_dim == 3:
        norm, generated_noise = norm.unsqueeze(1), generated_noise.unsqueeze(1)  # assuming ch dim is skipped  在第一维度增加维度
    anom = norm.clone()

    # invert noise for bright regions (bright regions are considered being on average > brightness_threshold)
    # 反转亮区噪声(亮区被认为是平均 > brightness_threshold)
    generated_noise = generated_noise.int()
    bright_regions = norm.sum(1) > brightness_threshold * norm.shape[1]
    for ch in range(norm.shape[1]):
        gnch = generated_noise[:, ch]
        gnch[bright_regions] = gnch[bright_regions] * -1
        generated_noise[:, ch] = gnch

    anom = (anom.int() + generated_noise).clamp(0, 255).byte()
    data = torch.cat((norm, anom))
    targets = torch.cat(
        (torch.ones(norm.size(0)) * nom_class,
         torch.ones(anom.size(0)) * outlier_classes[0])
    )
    if norm_dim == 3:
        data = data.squeeze(1)
    train_set.data = data
    train_set.targets = targets
    if gt:
        gtmaps = torch.cat(
            (torch.zeros_like(norm)[:, 0].float(),  # 0 for nominal
             (norm != anom).max(1)[0].clone().float())  # 1 for anomalous
        )
        if norm_dim == 4:
            gtmaps = gtmaps.unsqueeze(1)
        return train_set, gtmaps
    else:
        return train_set
