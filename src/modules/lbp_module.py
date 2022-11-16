import torch
import torch.nn.functional as torch_func


class LBPModule(torch.nn.Module):
    def __init__(self, input_shape: list[int], kernel_size: int, output_channel=1, pool_size=2):
        assert kernel_size % 2 == 1, "size must be an odd number"

        super().__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.input_shape = input_shape

        # 绕中心像素旋转的轴，轴两端所在的像素点坐标
        idx_pairs = [[[0, i], [self.kernel_size - 1, self.kernel_size - 1 - i]] for i in range(self.kernel_size)] + \
                    [[[i, self.kernel_size - 1], [self.kernel_size - 1 - i, 0]] for i in range(1, self.kernel_size - 1)]

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

        self.conv_final = torch.nn.Conv2d(in_channels=kernel_size * 2 - 2, out_channels=output_channel, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = torch_func.avg_pool2d(input=x, kernel_size=self.pool_size, stride=1, padding=int(self.pool_size / 2))
        x = torch_func.conv2d(x, self.lbp_kernel, padding=int(self.kernel_size / 2))
        x = x[:, ::2] - x[:, 1::2]

        x = self.conv_final(x)
        x = torch_func.layer_norm(x, normalized_shape=x.shape[-2:])
        x = torch_func.leaky_relu(x)

        return x


if __name__ == '__main__':
    i = torch.randint(0, 9, size=(2, 2, 64, 64)).float()
    # i = torch.arange(0, 2 * 4 * 4, dtype=torch.float).reshape((2, 1, 4, 4))
    print(i)
    module = LBPModule([2, 2, 64, 64], 3, 1, 3)
    o = module(i)
    print(o)
    print(o.shape)
