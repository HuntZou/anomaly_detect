import math
import random
import traceback
from itertools import cycle
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import utils
from datasets.bases import TorchvisionDataset
from datasets.outlier_exposure.mvtec import OEMvTec
from datasets.preprocessing import ImgGTTargetTransform


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
                    (1,) + ds.raw_shape, ds.normal_classes, limit_var=oe_limit,
                    logger=ds.logger, root=ds.root
                ).data_loader()
            )
        elif noise_mode == 'mvtec_gt':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1,) + ds.raw_shape, ds.normal_classes, limit_var=oe_limit,
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

            img_fg = utils.gen_img_fg(img)
            # img_fg = np.ones(img.shape[1:])

            supervise_mode = self.supervise_mode
            if random.random() < 0.9:
                if random.random() < 0.3:
                    """
                    添加cutpaste噪声
                    """
                    img, gt, target = self.CutPasteScar(
                        img, gt, target, self.ds, img_fg=img_fg
                    )
                else:
                    # 生成柏林噪声图案
                    img, gt, target = utils.gen_perlin_noise(img, img_fg)
            else:
                """
                添加随机颜色块噪声，生成一张和原图大小一样的带有很多色块的噪声图，加到原图上
                """
                img = img.unsqueeze(0) if img is not None else img  # 在第0维度增加维度
                # gt value 1 will be put to anom_label in mvtec_bases get_item
                gt = gt.unsqueeze(0).unsqueeze(0).fill_(1).float() if gt is not None else gt  # 在第0维度增加两个维度 gt张量所有值填充为1.0
                if self.noise_sampler is None:
                    generated_noise = self.ds._generate_noise(
                        self.noise_mode, img.shape, img_fg=img_fg
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
                        img, gt, target, self.ds, generated_noise, use_gt=True, invert_threshold=self.invert_threshold
                    )
                else:
                    raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
                img = img.squeeze(0) if img is not None else img
                img = (img - img.min()) / (img.max() - img.min())
                gt = gt.squeeze(0).squeeze(0) if gt is not None else gt

        # Image.fromarray(np.array(255*((img.permute(1, 2, 0) - img.min()) / (img.max() - img.min()))).astype(np.uint8)).save(os.path.join(utils.get_dir(TrainConfigures.output_dir, 'pseudo_mask'), f'{int(time.time() * 1000)}.png'))
        # Image.fromarray(np.array(img_fg).astype(np.uint8)*255).save(os.path.join(utils.get_dir(TrainConfigures.output_dir, 'pseudo_mask'), f'{int(time.time() * 1000)}_1.png'))
        # Image.fromarray(np.array(gt).astype(np.uint8)*255).save(os.path.join(utils.get_dir(TrainConfigures.output_dir, 'pseudo_mask'), f'{int(time.time() * 1000)}_2.png'))

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
        diff = ((anom.int() + generated_noise).clamp(0, 255) - anom.int())  # 对噪声的上界进行限制
        diff = diff.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()  # 求要添加的噪声均值
        diffi = ((anom.int() - generated_noise).clamp(0, 255) - anom.int())
        diffi = diffi.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()
        inv = [i for i, (d, di) in enumerate(zip(diff, diffi)) if d < invert_threshold and di > d]
        generated_noise[inv] = -generated_noise[inv]  # 如果可添加的噪声下界绝对值大于上界，并且噪声上界绝对值大于某个阈值，则添加负的噪声（图片变暗）

        anom = (anom.int() + generated_noise).clamp(0, 255).byte()  # 原始图片与噪声相加

        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied

        if use_gt:
            img = img.squeeze()
            anom = anom.squeeze()
            gt = (img != anom).max(0)[0].clone().float()  # ???
            # gt = gt.unsqueeze(1)  # value 1 will be put to anom_label in mvtec_bases get_item
        return anom, gt, t

    # 修改
    # def __CutPasteNormal(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset, area_ratio=[0.02,0.06],
    #                    aspect_ratio=0.3, use_gt: bool = False):
    def __CutPasteNormal(self, img: torch.Tensor,
                         img_fg,
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
        for i in range(random.randint(0, 2)):
            # ratio between area_ratio[0] and area_ratio[1]
            ratio_area = random.uniform(area_ratio[0], area_ratio[1]) * w * h

            # sample in log space
            log_ratio = torch.log(torch.tensor((aspect_ratio, 1 / aspect_ratio)))  # 转换为log空间
            aspect = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()  # 取出单元素张量的元素值，并返回该值，数值类型(如‘整形’)不变

            cut_w = max(int(round(math.sqrt(ratio_area * aspect)) * (np.sum(img_fg) / (img_fg.shape[0] * img_fg.shape[1]))), 1)  # 剪切的宽
            cut_h = max(int(round(math.sqrt(ratio_area / aspect)) * (np.sum(img_fg) / (img_fg.shape[0] * img_fg.shape[1]))), 1)  # 剪切的高

            # 只从前景部分取patch
            probs = img_fg.flatten() / np.sum(img_fg)
            index = np.random.choice(len(probs), p=probs)
            from_location_w = index % img_fg.shape[1]
            from_location_h = index // img_fg.shape[1]

            box = (from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h)
            patch = img.crop(box)
            patch = colorJitter(patch)
            # patch = mytransform(patch)

            # if random.random() < 0.5:
            #     sigma = np.random.uniform(0.1, 2.0)
            #     patch = patch.filter(ImageFilter.GaussianBlur(radius=sigma))

            rotation = [-45, 45]
            rot_deg = random.uniform(rotation[0], rotation[1])
            patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

            # 只粘贴到前景部分
            index = np.random.choice(len(probs), p=probs)
            to_location_w = index % img_fg.shape[1]
            to_location_h = index // img_fg.shape[1]

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

    def CutPasteScar(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset, img_fg, width=[1, 10], height=[10, 40], rotation=[-45, 45]):
        # cutPasteScar是必须的，但也会有一定概率同时出现cutPasteNormal
        if random.random() < 0.75:
            augmented = self.__CutPasteNormal(img, img_fg)
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

        if flag:
            augmented = img.copy()
        # augmented = img.copy()

        for i in range(random.randint(2, 4)):
            # cut region
            cut_w = random.uniform(width[0], width[1])
            cut_h = random.uniform(height[0], height[1])

            # 只剪切到前景区域
            probs = img_fg.flatten() / np.sum(img_fg)
            index = np.random.choice(len(probs), p=probs)
            from_location_w = index % img_fg.shape[1]
            from_location_h = index // img_fg.shape[1]

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
            rot_deg = random.uniform(rotation[0], rotation[1])
            patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

            # 只粘贴到前景区域
            index = np.random.choice(len(probs), p=probs)
            to_location_w = index % img_fg.shape[1]
            to_location_h = index // img_fg.shape[1]

            mask = patch.split()[-1]
            patch = patch.convert("RGB")

            augmented = augmented.copy()
            augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        # 这里会PIL的图像转换为tensor，注意，会将原来 [0,255] 转换到 [0.0,1.0]
        transform1 = transforms.ToTensor()
        augmented = transform1(augmented)
        img = transform1(img)

        # t = torch.IntTensor([1])        # t = t[0]
        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label
        # t = 1

        gt = (img != augmented).max(0)[0].clone().float()

        # Image.fromarray(np.array((augmented.permute(1, 2, 0) - augmented.min()) * 255 / (augmented.max() - augmented.min())).astype(np.uint8)).save(os.path.join(r'D:\Tmp\new\screw', f'{int(time.time()*1000)}.png'))
        return augmented, gt, t


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
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
        img[img > 255] = 255  # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
