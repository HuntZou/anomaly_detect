import random
import traceback
from itertools import cycle
from typing import List, Tuple

import numpy as np
import torch
from datasets.bases import TorchvisionDataset
from datasets.outlier_exposure.mvtec import OEMvTec
from datasets.preprocessing import ImgGTTargetTransform

from PIL import Image
import math
import torchvision.transforms as transforms


class OnlineSupervisor(ImgGTTargetTransform):
    invert_threshold = 0.025

    def __init__(self, ds: TorchvisionDataset, supervise_mode: str, noise_mode: str, oe_limit: int = np.infty,
                 p: float = 0.5, exclude: List[str] = ()):
        """
        This class is used as a Transform parameter for torchvision datasets.
        这个类用作torchvision数据集的Transform参数。
        During training it randomly replaces a sample of the dataset retrieved via the get_item method
        by an artificial anomaly.
        在训练期间，它将通过get_item方法检索到的数据集样本随机替换为一个人工异常。
        :param ds: some AD dataset for which the OnlineSupervisor is used. 一些使用OnlineSupervisor的AD数据集。
        :param supervise_mode: the type of artificial anomalies to be generated during training. 训练过程中产生的人工异常的类型。
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
            In addition to the offline noise modes, the OnlineSupervisor offers Outlier Exposure with MVTec-AD.
            除了离线噪声模式，OnlineSupervisor还提供了MVTec-AD的离群曝光。
            The oe_limit parameter for MVTec-AD limits the number of different samples per defection type
            MVTec-AD的oe_limit参数限制了每种缺陷类型的不同样本数量
            (including "good" instances, i.e. nominal ones in the test set).
        :param oe_limit: the number of different Outlier Exposure samples used in case of outlier exposure based noise.
        用于基于噪声的离群值暴露的不同离群值暴露样本的数量。
        :param p: the chance to replace a sample from the original dataset during training.
        在训练期间从原始数据集替换样本的机会。
        :param exclude: all class names that are to be excluded in Outlier Exposure datasets.
        在Outlier Exposure数据集中要排除的所有类名。
        """
        self.ds = ds  # 使用OnlineSupervisor的AD数据集。
        self.supervise_mode = supervise_mode  # 训练过程中产生人工异常的类型
        self.noise_mode = noise_mode
        self.oe_limit = oe_limit
        self.p = p
        self.noise_sampler = None

        if noise_mode == 'mvtec':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, ds.normal_classes, limit_var=oe_limit,
                    logger=ds.logger, root=ds.root
                ).data_loader()
            )
        elif noise_mode == 'mvtec_gt':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, ds.normal_classes, limit_var=oe_limit,
                    logger=ds.logger, gt=True, root=ds.root
                ).data_loader()
            )

    def __call__(self, img: torch.Tensor, gt: torch.Tensor, target: int,
                 replace: bool = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Based on the probability defined in __init__, replaces (img, gt, target) with an artificial anomaly.
        根据 __init__ 中定义的概率，用人工异常替换 (img, gt, target)。
        :param img: some torch tensor image
        :param gt: some ground-truth map (can be None)
        :param target: some label
        :param replace: whether to force or forbid a replacement, ignoring the probability.是否强制或禁止替换，忽略概率。
            The probability is only considered if replace == None.仅在 replace == None 时才考虑概率
        :return: (img, gt, target)
        """
        active = self.supervise_mode not in ['other', 'unsupervised']
        if active and (replace or replace is None and random.random() < self.p):
            supervise_mode = self.supervise_mode
            r = random.random()
            #修改
            # if supervise_mode in ['cutpaste']:
            # if random.random() < 0.35:
            #     img, gt, target = self.__CutPasteNormal(
            #         img, gt, target, self.ds
            #     )
            if random.random() < 0.85:
            # else:
                img, gt, target = self.CutPasteScar(
                    img, gt, target, self.ds
                )
            else:
                img = img.unsqueeze(0) if img is not None else img  # 在第0维度增加维度
                # gt value 1 will be put to anom_label in mvtec_bases get_item
                gt = gt.unsqueeze(0).unsqueeze(0).fill_(1).float() if gt is not None else gt  # 在第0维度增加两个维度 gt张量所有值填充为1.0
                if self.noise_sampler is None:
                    generated_noise = self.ds._generate_noise(
                        self.noise_mode, img.shape
                    )
                else:
                    try:
                        generated_noise = next(self.noise_sampler)
                    except RuntimeError:
                        generated_noise = next(self.noise_sampler)
                        self.ds.logger.warning(
                            'Had to resample in online_supervisor __call__ next(self.noise_sampler) because of {}'
                            .format(traceback.format_exc())
                        )
                    if isinstance(generated_noise, (tuple, list)):
                        generated_noise, gt = generated_noise
                if supervise_mode in ['noise']:
                    img, gt, target = self.__noise(img, gt, target, self.ds, generated_noise)
                elif supervise_mode in ['malformed_normal']:
                    img, gt, target = self.__malformed_normal(
                        img, gt, target, self.ds, generated_noise, invert_threshold=self.invert_threshold
                    )
                elif supervise_mode in ['malformed_normal_gt']:
                    img, gt, target = self.__malformed_normal(
                        img, gt, target, self.ds, generated_noise, use_gt=True,
                        invert_threshold=self.invert_threshold
                    )
                else:
                    raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
                img = img.squeeze(0) if img is not None else img
                gt = gt.squeeze(0).squeeze(0) if gt is not None else gt
        return img, gt, target

    def __noise(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset,
                generated_noise: torch.Tensor, use_gt: bool = False):
        if use_gt:
            raise ValueError('No GT mode for pure noise available!')
        anom = generated_noise.clamp(0, 255).byte()
        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied
        return anom, gt, t

    def __malformed_normal(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset,
                           generated_noise: torch.Tensor, use_gt: bool = False, invert_threshold: float = 0.025):
        assert (img.dim() == 4 or img.dim() == 3) and generated_noise.shape == img.shape
        anom = img.clone()

        # invert noise if difference of malformed and original is less than threshold and inverted difference is higher
        diff = ((anom.int() + generated_noise).clamp(0, 255) - anom.int())
        diff = diff.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()
        diffi = ((anom.int() - generated_noise).clamp(0, 255) - anom.int())
        diffi = diffi.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()
        inv = [i for i, (d, di) in enumerate(zip(diff, diffi)) if d < invert_threshold and di > d]
        generated_noise[inv] = -generated_noise[inv]

        anom = (anom.int() + generated_noise).clamp(0, 255).byte()  # 就是相加

        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied

        if use_gt:
            img = img.squeeze()
            anom = anom.squeeze()
            gt = (img != anom).max(0)[0].clone().float()
            # gt = gt.unsqueeze(1)  # value 1 will be put to anom_label in mvtec_bases get_item
        return anom, gt, t

    # 修改
    # def __CutPasteNormal(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset, area_ratio=[0.02,0.06],
    #                    aspect_ratio=0.3, use_gt: bool = False):
    def __CutPasteNormal(self, img: torch.Tensor,
                         area_ratio=[0.02, 0.06],
                         aspect_ratio=0.3, use_gt: bool = False):

        h = img.shape[1]  # 获取图像的高和宽  img为PIL图像
        w = img.shape[2]
        img = Image.fromarray(img.transpose(0, 2).transpose(0, 1).numpy(), mode='RGB')

        # colorJitter
        colorJitter = transforms.ColorJitter(brightness=0.1,  # 改变图像的属性：亮度、对比度。。。。
                                             contrast=0.1,
                                             saturation=0.1,
                                             hue=0.1)

        mytransform = AddSaltPepperNoise(0.02)

        augmented = img.copy()
        count = random.randint(1, 2)
        for i in range(0, 1):

            # ratio between area_ratio[0] and area_ratio[1]
            ratio_area = random.uniform(area_ratio[0],
                                        area_ratio[1]) * w * h  # random.uniform(a,b)输出[a,b]之间的随机浮点数

            # sample in log space
            log_ratio = torch.log(torch.tensor((aspect_ratio, 1 / aspect_ratio)))  # 转换为log空间
            aspect = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()  # 取出单元素张量的元素值，并返回该值，数值类型(如‘整形’)不变

            cut_w = int(round(math.sqrt(ratio_area * aspect)))  # 剪切的宽
            cut_h = int(round(math.sqrt(ratio_area / aspect)))  # 剪切的高

            # one might also want to sample from other images. currently we only sample from the image itself  目前只从图像本身取样
            from_location_h = int(random.uniform(0, h - cut_h))  # 随机选择裁剪的初始点
            from_location_w = int(random.uniform(0, w - cut_w))
            #
            # from_location_h = int(random.uniform(h / 5, 4 * h / 5 - cut_h))  # 随机选择裁剪的初始点
            # from_location_w = int(random.uniform(w / 5, 4 * w / 5 - cut_w))

            box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]  # 裁剪框
            patch = img.crop(box)
            patch = colorJitter(patch)
            # patch = mytransform(patch)


            # if random.random() < 0.5:
            #     sigma = np.random.uniform(0.1, 2.0)
            #     patch = patch.filter(ImageFilter.GaussianBlur(radius=sigma))

            rotation = [-45, 45]
            rot_deg = random.uniform(rotation[0], rotation[1])
            patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

            # paste
            to_location_h = int(random.uniform(0, h - patch.size[0]))
            to_location_w = int(random.uniform(0, w - patch.size[1]))
            #
            # to_location_h = int(random.uniform(h / 5, 4 * h / 5 - patch.size[0]))
            # to_location_w = int(random.uniform(w / 5, 4 * w / 5 - patch.size[1]))

            mask = patch.split()[-1]
            patch = patch.convert("RGB")

            # to_location_h = int(random.uniform(0, h - cut_h))
            # to_location_w = int(random.uniform(0, w - cut_w))

            # insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
            augmented = img.copy()
            # augmented.paste(patch, insert_box)  # 粘贴到图像的随机位置
            augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        # transform1 = transforms.ToTensor()
        # augmented = transform1(augmented)
        # img = transform1(img)
        #
        # t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label
        # # t = 1
        # # gt = torch.ones_like(img)[0].mul(255).byte()
        # gt = (img != augmented).max(0)[0].clone().float()
        #
        # return augmented, gt, t

        return augmented

    def CutPasteScar(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset, width=[2,20], height=[10,50], rotation=[-45,45]):
        if random.random() < 0.75:
            augmented = self.__CutPasteNormal(img)
            flag = False
        else:
            flag = True

        h = img.shape[1]  # 获取图像的高和宽  img为PIL图像
        w = img.shape[2]
        img = Image.fromarray(img.transpose(0, 2).transpose(0, 1).numpy(), mode='RGB')

        # colorJitter
        colorJitter = transforms.ColorJitter(brightness=0.1,  # 改变图像的属性：亮度、对比度。。。。
                                             contrast=0.1,
                                             saturation=0.1,
                                             hue=0.1)

        mytransform = AddSaltPepperNoise(0.02)

        if flag:
            augmented = img.copy()
        # augmented = img.copy()


        count = random.randint(2, 6)

        for i in range(0, count):
            # cut region
            cut_w = random.uniform(width[0], width[1])
            cut_h = random.uniform(height[0], height[1])

            from_location_h = int(random.uniform(0, h - cut_h))
            from_location_w = int(random.uniform(0, w - cut_w))
            #
            # from_location_h = int(random.uniform(h / 5, 4 * h / 5 - cut_h))
            # from_location_w = int(random.uniform(w / 5, 4 * w / 5 - cut_w))


            box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
            patch = img.crop(box)
            patch = colorJitter(patch)
            # if random.random()<0.5:
            #     patch = colorJitter(patch)

            # patch = mytransform(patch)

            # if random.random() < 0.5:
            #     sigma = np.random.uniform(0.1, 2.0)
            #     patch = patch.filter(ImageFilter.GaussianBlur(radius=sigma))


            # rotate
            rot_deg = random.uniform(rotation[0],rotation[1])
            patch = patch.convert("RGBA").rotate(rot_deg, expand=True)


            # paste
            to_location_h = int(random.uniform(0, h - patch.size[0]))
            to_location_w = int(random.uniform(0, w - patch.size[1]))
            # to_location_h = int(random.uniform(h / 5, 4 * h / 5 - patch.size[0]))
            # to_location_w = int(random.uniform(w / 5, 4 * w / 5 - patch.size[1]))

            mask = patch.split()[-1]
            patch = patch.convert("RGB")


            augmented = augmented.copy()
            augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        transform1 = transforms.ToTensor()
        augmented = transform1(augmented)
        img = transform1(img)

        # t = torch.IntTensor([1])        # t = t[0]
        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label
        # t = 1

        gt = (img != augmented).max(0)[0].clone().float()


        return augmented, gt, t





class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):

        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
