import torch
import torch.nn.functional as torch_func
from config import TrainConfigures


class LBPModule(torch.nn.Module):
    def __init__(self, input_shape: list[int], kernel_size: int, output_channel=1, pool_size=2):
        assert kernel_size % 2 == 1, "size must be an odd number"

        super().__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.input_shape = input_shape

        # 绕中心像素旋转的轴，轴两端所在的像素点坐标（固定只有四个轴方向）
        idx_pairs = [
            [[0, 0], [self.kernel_size - 1, self.kernel_size - 1]],
            [[0, int(self.kernel_size / 2)], [self.kernel_size - 1, int(self.kernel_size / 2)]],
            [[0, self.kernel_size - 1], [self.kernel_size - 1, 0]],
            [[int(self.kernel_size / 2), self.kernel_size - 1], [int(self.kernel_size / 2), 0]]
        ]

        # 根据轴两端的坐标得到两个卷积核
        def get_k(idx: []):
            kernel1 = torch.zeros(size=(input_shape[-3], self.kernel_size, self.kernel_size), device=TrainConfigures.device)
            kernel1[:, int(self.kernel_size / 2), int(self.kernel_size / 2)] = 1.
            kernel1[:, idx[0][0], idx[0][1]] = -1.

            kernel2 = torch.zeros(size=(input_shape[-3], self.kernel_size, self.kernel_size), device=TrainConfigures.device)
            kernel2[:, int(self.kernel_size / 2), int(self.kernel_size / 2)] = 1.
            kernel2[:, idx[1][0], idx[1][1]] = -1.
            return kernel1.unsqueeze(0), kernel2.unsqueeze(0)

        self.lbp_kernel = torch.cat([torch.cat(get_k(idx)) for idx in idx_pairs])

        self.conv_expand_idx = torch.nn.Conv2d(in_channels=3, out_channels=56, kernel_size=1)
        self.conv = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1)
        self.conv_final = torch.nn.Conv2d(in_channels=16, out_channels=output_channel, kernel_size=3, padding=1)

        self.clamp = ClampModel()

        self.quant_len = 8

        self.idxs = torch.cartesian_prod(torch.arange(0, self.quant_len, device=TrainConfigures.device), torch.arange(0, self.quant_len, device=TrainConfigures.device))
        masks = torch.zeros([self.quant_len ** 2, self.quant_len, self.quant_len], device=TrainConfigures.device)
        masks[torch.cat([torch.arange(0, self.quant_len ** 2, device=TrainConfigures.device).reshape(1, self.quant_len ** 2), self.idxs.permute([1, 0])]).tolist()] = 1
        self.masks = masks.unsqueeze(1).repeat([1, input_shape[-1] * input_shape[-2], 1, 1]).float()

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # 池化降噪
        x = torch_func.avg_pool2d(input=x, kernel_size=self.pool_size, stride=1, padding=int(self.pool_size / 2))
        # 计算中心像素和其邻域像素的各个梯度
        x = torch_func.conv2d(x, self.lbp_kernel, padding=int(self.kernel_size / 2))
        # 截断特征图，避免值域过大
        x = self.clamp(x)

        # 量阶--------------start
        # 第二维中的2表示以中心点为对称轴的两个点，后面计算类灰度共生矩阵时使用的就是这两个点作为表的横纵坐标
        x = x.reshape([batch_size * 4, 2, x.shape[-2], x.shape[-1]])

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
        # 将量阶的阶合并到统计数量中去，这里应该想办法直接计算measure后倒数第二维的笛卡尔积，但每找到直接计算的方法，曲线救国
        levels = self.idxs.unsqueeze(0).repeat([x.shape[0], 1, 1]) + 1
        levels = (levels.permute([0, 2, 1]) * measure[:, 1:2, :].permute([0, 2, 1])).permute([0, 2, 1])
        x = torch.cat([levels, x], dim=2)

        x = x.reshape([batch_size, 4, self.quant_len ** 2, 3])
        # 量阶--------------end

        # 最后接两层卷积
        x = x.permute(0, 3, 1, 2)
        x = torch_func.leaky_relu(self.conv_expand_idx(x))
        x = x.permute(0, 2, 3, 1)
        x = torch_func.interpolate(x, self.input_shape[-2:])
        x = torch_func.layer_norm(x, normalized_shape=x.shape[-2:])
        x = torch_func.leaky_relu(self.conv(x))
        x = torch_func.leaky_relu(self.conv_final(x))
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
