import torch
from datasets.noise import confetti_noise, colorize_noise, solid, smooth_noise
# from datasets.outlier_exposure.cifar100 import OECifar100
# from datasets.outlier_exposure.emnist import OEEMNIST
# from datasets.outlier_exposure.imagenet import OEImageNet, OEImageNet22k
from logging import Logger

MODES = [
    'gaussian', 'uniform', 'blob', 'mixed_blob', 'solid', 'confetti',  # Synthetic Anomalies  合成异常
    'imagenet', 'imagenet22k', 'cifar100', 'emnist',  # Outlier Exposure
    'mvtec', 'mvtec_gt'  # Outlier Exposure online supervision only
]


def generate_noise(noise_mode: str, size: torch.Size, oe_limit: int,
                   logger: Logger = None, datadir: str = None) -> torch.Tensor:
    """
    Given a noise_mode, generates noise images.  生成相同大小的noise图像
    :param noise_mode: one of the available noise_nodes, see MODES:
        'gaussian': choose pixel values based on Gaussian distribution.
        根据高斯分布选择像素值
        'uniform: choose pixel values based on uniform distribution.
        根据均匀分布选择像素值
        'blob': images of randomly many white rectangles of random size.
        图像随机许多白色矩形的大小随机。
        'mixed_blob': images of randomly many rectangles of random size, approximately half of them
            are white and the others have random colors per pixel.
        随机大小的矩形的图像，大约是它们的一半是白色的，其他的每个像素有随机的颜色。
        'solid': images of solid colors, i.e. one random color per image.
        纯色图像，即每个图像一个随机颜色。
        'confetti': confetti noise as seen in the paper. Random size, orientation, and number.
        纸上看到的五彩纸屑噪音。大小、方向和数字都是随机的。
            They are also smoothened. Half of them are white, the rest is of one random color per rectangle.
            它们也是平滑的。其中一半是白色的，其余的是每个矩形的随机颜色。
        'imagenet': Outlier Exposure with ImageNet.使用ImageNet曝光
        'imagenet22k': Outlier Exposure with ImageNet22k, i.e. the full release fall 2011.
        'cifar100': Outlier Exposure with Cifar-100.
        'emnist': Outlier Exposure with EMNIST.
    :param size: number and size of the images (n x c x h x w)
    :param oe_limit: limits the number of different samples for Outlier Exposure
    :param logger: some logger
    :param datadir: the root directory of datsets (for Outlier Exposure)
    :return: a torch tensor of noise images
    """
    if noise_mode is not None:
        if noise_mode in ['gaussian']:
            generated_noise = (torch.randn(size) * 64)
        elif noise_mode in ['uniform']:
            generated_noise = (torch.rand(size)).mul(255)
        elif noise_mode in ['blob']:
            generated_noise = confetti_noise(size, 0.002, (6, 6), fillval=255, clamp=False, awgn=0)
        elif noise_mode in ['mixed_blob']:
            generated_noise_rgb = confetti_noise(size, 0.00001, (34, 34), fillval=255, clamp=False, awgn=0)
            generated_noise = confetti_noise(size, 0.00001, (34, 34), fillval=255, clamp=False, awgn=0)
            generated_noise_rgb = colorize_noise(generated_noise_rgb, (0, 0, 0), (255, 255, 255), p=1)
            generated_noise = generated_noise_rgb + generated_noise
        elif noise_mode in ['solid']:
            generated_noise = solid(size)
        elif noise_mode in ['confetti']:
            generated_noise_rgb = confetti_noise(
                size, 0.000018, ((8, 8), (54, 54)), fillval=255, clamp=False, awgn=0, rotation=45, colorrange=(-256, 0)
            )
            generated_noise = confetti_noise(
                size, 0.000012, ((8, 8), (54, 54)), fillval=-255, clamp=False, awgn=0, rotation=45
            )
            generated_noise = generated_noise_rgb + generated_noise
            generated_noise = smooth_noise(generated_noise, 25, 5, 1.0)
        # elif noise_mode in ['imagenet']:
        #     generated_noise = next(iter(OEImageNet(
        #         size, limit_var=oe_limit, root=datadir
        #     ).data_loader()))
        # elif noise_mode in ['imagenet22k']:
        #     generated_noise = next(iter(
        #         OEImageNet22k(size, limit_var=oe_limit, logger=logger, root=datadir).data_loader()
        #     ))
        # elif noise_mode in ['cifar100']:
        #     generated_noise = next(iter(OECifar100(
        #         size, limit_var=oe_limit, root=datadir
        #     ).data_loader()))
        # elif noise_mode in ['emnist']:
        #     generated_noise = next(iter(OEEMNIST(
        #         size, limit_var=oe_limit, root=datadir
        #     ).data_loader()))
        elif noise_mode in ['mvtec', 'mvtec_gt']:
            raise NotImplementedError(
                'MVTec-AD and MVTec-AD with ground-truth maps is only available with online supervision.'
            )
        else:
            raise NotImplementedError('Supervise noise mode {} unknown (offline version).'.format(noise_mode))
        return generated_noise
