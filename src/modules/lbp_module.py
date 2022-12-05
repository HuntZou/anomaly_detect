import torch
import torch.nn.functional as torch_func


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
            kernel1 = torch.zeros(size=(input_shape[-3], self.kernel_size, self.kernel_size)).cuda()
            kernel1[:, int(self.kernel_size / 2), int(self.kernel_size / 2)] = 1.
            kernel1[:, idx[0][0], idx[0][1]] = -1.

            kernel2 = torch.zeros(size=(input_shape[-3], self.kernel_size, self.kernel_size)).cuda()
            kernel2[:, int(self.kernel_size / 2), int(self.kernel_size / 2)] = 1.
            kernel2[:, idx[1][0], idx[1][1]] = -1.
            return kernel1.unsqueeze(0), kernel2.unsqueeze(0)

        self.lbp_kernel = torch.cat([torch.cat(get_k(idx)) for idx in idx_pairs])

        self.conv_expand_idx = torch.nn.Conv2d(in_channels=3, out_channels=56, kernel_size=1)
        self.conv = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1)
        self.conv_final = torch.nn.Conv2d(in_channels=16, out_channels=output_channel, kernel_size=3, padding=1)

        self.clamp = ClampModel()

        self.quant_len = 8

        self.idxs = torch.cartesian_prod(torch.arange(0, self.quant_len), torch.arange(0, self.quant_len)).cuda()
        masks = torch.zeros([self.quant_len ** 2, self.quant_len, self.quant_len]).cuda()
        masks[torch.cat([torch.arange(0, self.quant_len ** 2).reshape(1, self.quant_len ** 2).cuda(), self.idxs.permute([1, 0])]).tolist()] = 1
        self.masks = masks.unsqueeze(1).repeat([1, input_shape[-1] * input_shape[-2], 1, 1]).float().cuda()

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # 池化降噪
        x = torch_func.avg_pool2d(input=x, kernel_size=self.pool_size, stride=1, padding=int(self.pool_size / 2))
        # 计算中心像素和其邻域像素的各个梯度
        x = torch_func.conv2d(x, self.lbp_kernel, padding=int(self.kernel_size / 2))
        # 截断特征图，避免值域过大
        x = self.clamp(x)

        # 量阶--------------start
        x = x.reshape([batch_size * 4, 2, x.shape[-2], x.shape[-1]])
        levels_0 = torch.stack([
            torch.vstack([
                *((x[i][0].max() - x[i][0].min()) * l / self.quant_len + x[i][0].min() for l in range(self.quant_len)),
                x[i][0].max()
            ]) for i in range(len(x))
        ], dim=0)

        levels_1 = torch.stack([
            torch.vstack([
                *((x[i][1].max() - x[i][1].min()) * l / self.quant_len + x[i][1].min() for l in range(self.quant_len)),
                x[i][1].max()]) for i in range(len(x))
        ], dim=0)

        x0 = x[:, 0, ...].reshape([x.shape[0], -1]).unsqueeze(dim=1).repeat([1, self.quant_len, 1])
        x0[(x0 - levels_0[:, :-1] < 0) | (x0 - levels_0[:, 1:] > 0)] = 0
        x0 = x0.unsqueeze(1).permute(0, 3, 2, 1)

        x1 = x[:, 1, ...].reshape([x.shape[0], -1]).unsqueeze(dim=1).repeat([1, self.quant_len, 1])
        x1[(x1 - levels_1[:, :-1] < 0) | (x1 - levels_1[:, 1:] > 0)] = 0
        x1 = x1.unsqueeze(1).permute(0, 3, 1, 2)

        x = x0.matmul(x1)

        x = torch.nn.functional.conv2d(x, self.masks) / torch.sum(x)
        x = x.squeeze(-1)
        levels = self.idxs.unsqueeze(0).repeat([x.shape[0], 1, 1])
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
