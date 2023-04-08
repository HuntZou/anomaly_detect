import math

import torch
import torch.nn.functional as torch_func

from config import TrainConfigures


class LBPModule(torch.nn.Module):
    def __init__(self, input_shape: list[int], kernel_size: int, output_channel=128, pool_size=3, quant_level=8, reduce_input_channel=4):
        assert kernel_size % 2 == 1, "size must be an odd number"

        super().__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.input_shape = input_shape
        self.reduce_input_channel = reduce_input_channel

        # 绕中心像素旋转的轴，轴两端所在的像素点坐标（固定只有四个轴方向）
        idx_pairs = [
            [[0, 0], [self.kernel_size - 1, self.kernel_size - 1]],
            [[0, int(self.kernel_size / 2)], [self.kernel_size - 1, int(self.kernel_size / 2)]],
            [[0, self.kernel_size - 1], [self.kernel_size - 1, 0]],
            [[int(self.kernel_size / 2), self.kernel_size - 1], [int(self.kernel_size / 2), 0]]
        ]

        # 根据轴两端的坐标得到两个卷积核
        def get_k(idx: []):
            kernel1 = torch.zeros(size=(1, self.kernel_size, self.kernel_size), device=TrainConfigures.device)
            kernel1[:, int(self.kernel_size / 2), int(self.kernel_size / 2)] = 1.
            kernel1[:, idx[0][0], idx[0][1]] = -1.

            kernel2 = torch.zeros(size=(1, self.kernel_size, self.kernel_size), device=TrainConfigures.device)
            kernel2[:, int(self.kernel_size / 2), int(self.kernel_size / 2)] = 1.
            kernel2[:, idx[1][0], idx[1][1]] = -1.
            return kernel1.unsqueeze(0), kernel2.unsqueeze(0)

        self.lbp_kernel = torch.cat([torch.cat(get_k(idx)) for idx in idx_pairs])

        self.reduce_channel = torch.nn.Conv2d(in_channels=self.input_shape[-3], out_channels=self.reduce_input_channel, kernel_size=3, padding=1)
        self.conv = torch.nn.Conv2d(in_channels=self.reduce_input_channel, out_channels=256, kernel_size=3, padding=1)
        self.conv_bn = torch.nn.BatchNorm2d(256)
        self.conv_final = torch.nn.Conv2d(in_channels=256, out_channels=output_channel, kernel_size=3, padding=1)
        self.conv_final_bn = torch.nn.BatchNorm2d(output_channel)

        self.clamp = ClampModel()

        self.quant_len = quant_level

        self.idxs = torch.cartesian_prod(torch.arange(0, quant_level, device=TrainConfigures.device), torch.arange(0, quant_level, device=TrainConfigures.device))
        masks = torch.zeros([quant_level ** 2, quant_level, quant_level], device=TrainConfigures.device)
        masks[torch.cat([torch.arange(0, quant_level ** 2, device=TrainConfigures.device).reshape(1, quant_level ** 2), self.idxs.permute([1, 0])]).tolist()] = 1
        self.masks = masks.unsqueeze(1).repeat([1, input_shape[-1] * input_shape[-2], 1, 1]).float()

        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), stride=1, padding=1)

        # 注意力机制设置隐藏状态特征长度
        self.attention_hidden_feature_len = 22 ** 2
        self.attention_embed_len = 8 ** 2

        self.reduce_attention_hidden_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=math.prod(input_shape[-2:]), out_channels=self.attention_hidden_feature_len, kernel_size=1),
            torch.nn.LeakyReLU()
        )

        self.gen_q = torch.nn.Sequential(
            torch.nn.Linear(in_features=3 * 3, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=256, out_features=self.attention_embed_len),
            torch.nn.LeakyReLU()
        )

        self.gen_k = torch.nn.Sequential(
            torch.nn.Linear(in_features=3 * 3, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=256, out_features=self.attention_embed_len),
            torch.nn.LeakyReLU()
        )

        self.gen_v = torch.nn.Sequential(
            torch.nn.Linear(in_features=4 * quant_level ** 2 * 3, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=256, out_features=self.attention_hidden_feature_len),
            torch.nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # 截断特征图，避免值域过大
        x = self.clamp(x)

        # 池化降噪
        x = torch_func.avg_pool2d(input=x, kernel_size=self.pool_size, stride=1, padding=int(self.pool_size / 2))

        # 原始特征图通道数较多，这里将其减少
        x = self.reduce_channel(x)

        # 原始特征图副本，用于与后面的结果concatenate
        x = x.reshape([batch_size * self.reduce_input_channel, 1, *self.input_shape[-2:]])
        x_after_pool = x.clone()

        # 计算中心像素和其邻域像素的各个梯度
        x = torch_func.conv2d(x, self.lbp_kernel, padding=int(self.kernel_size / 2))

        # 量阶--------------start
        # 第二维中的2表示以中心点为对称轴的两个点，后面计算类灰度共生矩阵时使用的就是这两个点作为表的横纵坐标
        x = x.reshape([batch_size * self.reduce_input_channel * 4, 2, x.shape[-2], x.shape[-1]])

        # 这两行代码计算量阶中每个阶的上下界
        common_difference = (x.amax(axis=(-1, -2)) - x.amin(axis=(-1, -2))) / self.quant_len
        measure = (torch.einsum('i,jk->ijk', torch.arange(self.quant_len + 1, device=TrainConfigures.device), common_difference) + x.amin(axis=(-1, -2))).permute(1, 0, 2)

        x0 = x[:, 0, ...].reshape([x.shape[0], -1]).unsqueeze(dim=1).repeat([1, self.quant_len, 1])
        x0[(x0 - measure[:, :-1, :1] < 0) | (x0 - measure[:, 1:, :1] > 0)] = 0
        x0 = x0.unsqueeze(1).permute(0, 3, 2, 1)

        x1 = x[:, 1, ...].reshape([x.shape[0], -1]).unsqueeze(dim=1).repeat([1, self.quant_len, 1])
        x1[(x1 - measure[:, :-1, 1:] < 0) | (x1 - measure[:, 1:, 1:] > 0)] = 0
        x1 = x1.unsqueeze(1).permute(0, 3, 1, 2)

        x = x0.matmul(x1)

        x = torch.nn.functional.conv2d(x, self.masks) / torch.sum(x)
        x = x.squeeze(-1)
        # 将量阶的阶合并到统计数量中去，这里应该想办法直接计算measure后倒数第二维的笛卡尔积，但没找到直接计算的方法，曲线救国
        levels = self.idxs.unsqueeze(0).repeat([x.shape[0], 1, 1]) + 1
        levels = (levels.permute([0, 2, 1]) * measure[:, 1:2, :].permute([0, 2, 1])).permute([0, 2, 1])
        x = torch.cat([levels, x], dim=2)

        x = x.reshape([batch_size * self.reduce_input_channel, 4, self.quant_len ** 2, 3])
        # 量阶--------------end

        # 将二阶统计量展平，然后使用MLP将其转换为长度等于特征图像素数量的一个context
        x = x.reshape([x.shape[0], -1])
        v = self.gen_v(x).unsqueeze(-1)

        # 将原本图像中的每个点都使用其周围的3*3的卷积核代替
        x_after_pool = self.unfold(x_after_pool).permute([0, 2, 1]).reshape([x_after_pool.shape[0], math.prod(self.input_shape[-2:]), 3, 3])
        q = self.gen_q(x_after_pool.reshape([*x_after_pool.shape[:-2], -1]))

        x_after_pool = self.reduce_attention_hidden_feature(x_after_pool)
        k = self.gen_k(x_after_pool.reshape([*x_after_pool.shape[:-2], -1]))

        w = torch.nn.functional.softmax(torch.bmm(q, k.permute([0, 2, 1])), dim=-1)
        x = torch.bmm(w, v)

        x = x.reshape([batch_size, self.reduce_input_channel, *self.input_shape[-2:]])

        # 最后接两层卷积
        x = torch_func.layer_norm(x, normalized_shape=x.shape[-2:])
        x = torch_func.leaky_relu((self.conv_bn(self.conv(x))))
        x = torch_func.leaky_relu(self.conv_final_bn(self.conv_final(x)))
        return x


class ClampModel(torch.nn.Module):
    def __init__(self):
        super(ClampModel, self).__init__()
        self.clamp_low_threshold = torch.nn.Parameter(data=torch.tensor(-150.), requires_grad=True)
        self.clamp_high_threshold = torch.nn.Parameter(data=torch.tensor(150.), requires_grad=True)

        self.clamp = Clamp().apply

    def forward(self, x):
        x = self.clamp(x, self.clamp_low_threshold, self.clamp_high_threshold)
        return x


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, low_threshold, high_threshold):
        return torch.clamp(x, low_threshold, high_threshold)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, torch.sum(grad_output), torch.sum(grad_output)
