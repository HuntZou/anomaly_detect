import torch
import torch.nn as nn
from torch.nn import functional as F
from lbp_module import LBPModule


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                 has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class QCO_1d(nn.Module):
    def __init__(self, level_num):
        super(QCO_1d, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(512, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num
        self.se = SELayer(128, reduction_ratio=8)
        self.ass = SSPCAB(128)
        self.cam = ChannelAttention(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x_ave = self.cam(x)
        N, C, H, W = x.shape
        x_ave = F.adaptive_avg_pool2d(x, (1, 1))  # size: NxCx1x1
        # x_max = F.adaptive_max_pool2d(x, (1, 1))
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x, dim=1)).sum(1)  # NxHxW
        # cos_sim_max = (F.normalize(x_max, dim=1) * F.normalize(x, dim=1)).sum(1)
        # cos_sim = cos_sim + cos_sim_max
        cos_sim = cos_sim.view(N, -1)  # NxHW
        cos_sim_min, _ = cos_sim.min(-1)  # 取出最小值 N
        cos_sim_min = cos_sim_min.unsqueeze(-1)  # 最小值 Nx1
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)  # 最大值 Nx1
        q_levels = torch.arange(self.level_num).float().cuda()  # 量化级数
        q_levels = q_levels.expand(N, self.level_num)  # 扩充为：NxL
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min  # L级数
        q_levels = q_levels.unsqueeze(1)  # Nx1xL
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)  # Nx1 级间间隔
        cos_sim = cos_sim.unsqueeze(-1)  # NxHWx1
        quant = 1 - torch.abs(q_levels - cos_sim)  # size:NxHWxL
        quant = quant * (quant > (1 - q_levels_inter))  # size: NxHWxL
        # quant = (quant > (1 - q_levels_inter)).float()
        sta = quant.sum(1)  # 计数 size: NxL 每个量级上像素的个数
        sta = sta / (sta.sum(-1).unsqueeze(-1))  # 计数频率size:NxL,每个量级上像素个数所占总像素的频率
        sta = sta.unsqueeze(1)  # Nx1xL
        sta = torch.cat([q_levels, sta], dim=1)  # Nx2xL
        sta = self.f1(sta)  # size:Nx64xL
        sta = self.f2(sta)  # size:Nx128xL
        # x_ave = x_ave + x_max
        x_ave = x_ave.squeeze(-1).squeeze(-1)  # size:NxC
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0)  # NxCxL
        # x_max = x_max.squeeze(-1).squeeze(-1)  # size:NxC
        # x_max = x_max.expand(self.level_num, N, C).permute(1, 2, 0)  # NxCxL
        sta = torch.cat([sta, x_ave], dim=1)  # Nx(128+C + C)xL
        sta = self.out(sta)  # Nx128xL
        return sta, quant  # Nx128xL  NxHWxL


class QCO_2d(nn.Module):
    def __init__(self, scale, level_num, n):
        super(QCO_2d, self).__init__()
        self.f1 = nn.Sequential(ConvBNReLU(3, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='2d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='2d')
        self.out = nn.Sequential(ConvBNReLU(384, 128, 1, 1, 0, has_bn=True, has_relu=True, mode='2d'), ConvBNReLU(128, 128, 1, 1, 0, has_bn=True, has_relu=False, mode='2d'))
        self.scale = scale
        self.level_num = level_num

    def forward(self, x):
        N1, C1, H1, W1 = x.shape
        if H1 // self.level_num != 0 or W1 // self.level_num != 0:
            x = F.adaptive_avg_pool2d(x, ((int(H1 / self.level_num) * self.level_num), int(W1 / self.level_num) * self.level_num))
        N, C, H, W = x.shape
        self.size_h = int(H / self.scale)
        self.size_w = int(W / self.scale)
        x_ave = F.adaptive_avg_pool2d(x, (self.scale, self.scale))
        x_ave_up = F.adaptive_avg_pool2d(x_ave, (H, W))
        # x_max = F.adaptive_max_pool2d(x, (self.scale, self.scale))
        # x_max_up = F.adaptive_avg_pool2d(x_max, (H, W))
        cos_sim = (F.normalize(x_ave_up, dim=1) * F.normalize(x, dim=1)).sum(1)  # NxHxW
        # cos_sim_max = (F.normalize(x_max_up, dim=1) * F.normalize(x, dim=1)).sum(1)
        # cos_sim = cos_sim + cos_sim_max
        # cos_sim =1 - (-torch.norm(x-x_ave_up, p=2, dim=1).unsqueeze(1)).exp()
        cos_sim = cos_sim.unsqueeze(1)  # Nx1xHxW
        cos_sim = cos_sim.reshape(N, 1, self.size_h, self.scale, self.size_w, self.scale)
        cos_sim = cos_sim.permute(0, 1, 2, 4, 3, 5)
        cos_sim = cos_sim.reshape(N, 1, int(self.size_h * self.size_w), int(self.scale * self.scale))
        # cos_sim = cos_sim.permute(0, 1, 3, 2)
        cos_sim = cos_sim.squeeze(1)
        cos_sim_min, _ = cos_sim.min(1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        q_levels = torch.arange(self.level_num).float().cuda()
        q_levels = q_levels.expand(N, self.scale * self.scale, self.level_num)
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(1).unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)
        q_levels = q_levels.unsqueeze(1)
        quant = 1 - torch.abs(q_levels - cos_sim)
        quant = quant * (quant > (1 - q_levels_inter))
        # quant = (quant > (1 - q_levels_inter)).float()
        quant = quant.view([N, self.size_h, self.size_w, self.scale * self.scale, self.level_num])
        quant = quant.permute(0, -2, -1, 1, 2)
        quant = quant.contiguous().view(N, -1, self.size_h, self.size_w)
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        quant = quant.view(N, self.scale * self.scale, self.level_num, self.size_h + 1, self.size_w + 1)
        quant_left = quant[:, :, :, :self.size_h, :self.size_w].unsqueeze(3)
        quant_right = quant[:, :, :, 1:, 1:].unsqueeze(2)
        # quant_right1 = quant[:, :, :, :self.size_h, 1:].unsqueeze(2)
        # quant_right2 = quant[:, :, :, 1:, :self.size_w].unsqueeze(2)
        quant = quant_left * quant_right
        # quant1 = quant_left * quant_right1
        # quant2 = quant_left * quant_right2
        sta = quant.sum(-1).sum(-1)
        sta = sta / (sta.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6)
        sta = sta.unsqueeze(1)
        q_levels = q_levels.expand(self.level_num, N, 1, self.scale * self.scale, self.level_num)
        q_levels_h = q_levels.permute(1, 2, 3, 0, 4)
        q_levels_w = q_levels_h.permute(0, 1, 2, 4, 3)
        sta = torch.cat([q_levels_h, q_levels_w, sta], dim=1)
        sta = sta.view(N, 3, self.scale * self.scale, -1)
        sta = self.f1(sta)
        sta = self.f2(sta)
        # x_ave = x_ave + x_max
        x_ave = x_ave.view(N, C, -1)
        x_ave = x_ave.expand(self.level_num * self.level_num, N, C, self.scale * self.scale)
        x_ave = x_ave.permute(1, 2, 3, 0)
        # x_max = x_max.view(N, C, -1)
        # x_max = x_max.expand(self.level_num * self.level_num, N, C, self.scale * self.scale)
        # x_max = x_max.permute(1, 2, 3, 0)
        sta = torch.cat([x_ave, sta], dim=1)
        sta = self.out(sta)
        sta = sta.mean(-1)
        sta = sta.view(N, sta.shape[1], self.scale, self.scale)
        return sta


class LBP(nn.Module):
    def __init__(self, level_num):
        super(LBP, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, has_relu=False),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.conv3 = ConvBNReLU(128, 256, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(
            ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
            nn.LeakyReLU(inplace=True)
        )
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out1 = ConvBNReLU(256, 256, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num
        self.k = ConvBNReLU(256, 256, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(256, 256, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(256, 256, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(256, 256, 1, 1, 0, mode='1d')

    def forward(self, x):
        print(f"x shape: {x.shape}")
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape
        x_ave = F.adaptive_avg_pool2d(x, (1, 1))  # size: NxCx1x1
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x, dim=1)).sum(1)  # NxHxW
        self.size_h = int(H / 3)  # 以scale为1的尺寸大小
        self.size_w = int(W / 3)
        cos_sim = F.adaptive_avg_pool2d(cos_sim, (self.size_h * 3, self.size_w * 3))
        print(f"cos_sim shape: {cos_sim.shape}")
        cos_sim = cos_sim.reshape(N, 3, self.size_h, 3, self.size_w)
        print(f"cos_sim shape: {cos_sim.shape}")
        cos_sim = cos_sim.permute(0, 1, 3, 2, 4)  # 分割成每一块
        cos_sim = cos_sim.reshape(N, 9, int(self.size_h * self.size_w))
        cos_sim = cos_sim.permute(0, 2, 1)  # Nxsize_h*size_wx9
        print(f"cos_sim shape: {cos_sim.shape}")
        N, H1, W1 = cos_sim.size()
        cos_sim = (cos_sim > cos_sim[:, :, 4].unsqueeze(-1).expand(N, H1, W1)).float()
        cos_sim = cos_sim[:, :, 0] + cos_sim[:, :, 1] * 2 + cos_sim[:, :, 2] * 4 + cos_sim[:, :, 3] * 8 + cos_sim[:, :, 5] * 16 + cos_sim[:, :, 6] * 32 + cos_sim[:, :, 7] * 64 + cos_sim[:, :, 8] * 128
        min = cos_sim.min(-1)[0].unsqueeze(-1).expand(N, H1)
        max = cos_sim.max(-1)[0].unsqueeze(-1).expand(N, H1)
        cos_sim = (cos_sim - min) / (max - min)
        cos_sim = cos_sim.float().cuda()
        # Nxsize_h*size_w
        cos_sim = cos_sim.unsqueeze(1).view(N, 1, -1)
        cos_sim = cos_sim.view(N, -1)
        cos_sim_min, _ = cos_sim.min(-1)  # 取出最小值 N
        cos_sim_min = cos_sim_min.unsqueeze(-1)  # 最小值 Nx1
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)  # 最大值 Nx1
        q_levels = torch.arange(self.level_num).float().cuda()  # 量化级数
        q_levels = q_levels.expand(N, self.level_num)  # 扩充为：NxL
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min  # L级数
        q_levels = q_levels.unsqueeze(1)  # Nx1xL
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)  # Nx1 级间间隔
        cos_sim = cos_sim.unsqueeze(-1)  # NxHWx1
        quant = 1 - torch.abs(q_levels - cos_sim)  # size:NxHWxL
        quant = quant * (quant > (1 - q_levels_inter))  # size: NxHWxL
        sta = quant.sum(1)  # 计数 size: NxL 每个量级上像素的个数
        sta = sta / (sta.sum(-1).unsqueeze(-1))  # 计数频率size:NxL,每个量级上像素个数所占总像素的频率
        sta = sta.unsqueeze(1)  # Nx1xL
        sta = torch.cat([q_levels, sta], dim=1)  # Nx2xL
        sta = self.f1(sta)  # size:Nx64xL
        sta = self.f2(sta)  # size:Nx128xL
        x_ave = x_ave.squeeze(-1).squeeze(-1)  # size:NxC
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0)  # NxCxL
        sta = torch.cat([sta, x_ave], dim=1)  # Nx(128+C)xL
        sta = self.out1(sta)  # Nx256xL

        k = self.k(sta)
        q = self.q(sta)  # Nx128xL
        v = self.v(sta)  # Nx128xL
        k = k.permute(0, 2, 1)  # sta:NxLx128
        w = torch.bmm(k, q)  # NxLxL
        w = F.softmax(w, dim=-1)  # NxLxL
        v = v.permute(0, 2, 1)  # NxLx128
        f = torch.bmm(w, v)  # NxLx128
        f = f.permute(0, 2, 1)  # Nx128xL
        f = self.out(f)
        quant = quant.permute(0, 2, 1)
        out = torch.bmm(f, quant)  # Nx256xH1W1
        out = out.view(N, 256, self.size_h, self.size_h)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        # out = self.conv3(out)
        return out  # Nx256xHxW


class TEM(nn.Module):
    def __init__(self, level_num):
        super(TEM, self).__init__()
        self.level_num = level_num
        self.qco = QCO_1d(level_num)
        self.k = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        N, C, H, W = x.shape
        sta, quant = self.qco(x)  # Nx128xL  NxHWxL
        k = self.k(sta)
        q = self.q(sta)  # Nx128xL
        v = self.v(sta)  # Nx128xL
        k = k.permute(0, 2, 1)  # sta:NxLx128
        w = torch.bmm(k, q)  # NxLxL
        w = F.softmax(w, dim=-1)  # NxLxL
        v = v.permute(0, 2, 1)  # NxLx128
        f = torch.bmm(w, v)  # NxLx128
        f = f.permute(0, 2, 1)  # Nx128xL
        f = self.out(f)  # Nx256xL
        quant = quant.permute(0, 2, 1)  # NxLxHW
        out = torch.bmm(f, quant)  # Nx256xHW
        out = out.view(N, 256, H, W)  # Nx256xHxW
        # out = self.dropout(out)
        return out


class PTFEM(nn.Module):
    def __init__(self):
        super(PTFEM, self).__init__()
        self.conv = ConvBNReLU(768, 256, 1, 1, 0, has_bn=False, has_relu=False)
        self.qco_1 = QCO_2d(1, 8, 1)
        self.qco_2 = QCO_2d(2, 8, 1)
        self.qco_3 = QCO_2d(4, 8, 1)
        self.qco_6 = QCO_2d(8, 8, 1)
        self.out = ConvBNReLU(512, 256, 1, 1, 0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.conv(x)
        sta_1 = self.qco_1(x)
        sta_2 = self.qco_2(x)
        sta_3 = self.qco_3(x)
        sta_6 = self.qco_6(x)

        N, C = sta_1.shape[:2]
        sta_1 = sta_1.view(N, C, 1, 1)
        sta_2 = sta_2.view(N, C, 2, 2)
        sta_3 = sta_3.view(N, C, 4, 4)
        sta_6 = sta_6.view(N, C, 8, 8)
        sta_1 = F.interpolate(sta_1, size=(H, W), mode='bilinear', align_corners=True)
        sta_2 = F.interpolate(sta_2, size=(H, W), mode='bilinear', align_corners=True)
        sta_3 = F.interpolate(sta_3, size=(H, W), mode='bilinear', align_corners=True)
        sta_6 = F.interpolate(sta_6, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat([sta_1, sta_2, sta_3, sta_6], dim=1)
        x = self.out(x)
        x = self.dropout(x)
        return x


class STL(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv_start = ConvBNReLU(in_channel, 512, 1, 1, 0)
        self.tem = TEM(128)
        self.ptfem = PTFEM()
        # self.lbp = LBP(256)
        self.lbp1 = LBPModule(input_shape=[512, 64, 64], kernel_size=3, output_channel=64, pool_size=1)
        self.lbp2 = LBPModule(input_shape=[512, 64, 64], kernel_size=7, output_channel=64, pool_size=3)
        self.lbp3 = LBPModule(input_shape=[512, 64, 64], kernel_size=11, output_channel=64, pool_size=5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv_start(x)
        x_lbp1 = self.lbp1(x)
        x_lbp2 = self.lbp2(x)
        x_lbp3 = self.lbp3(x)
        x_tem = self.tem(x)
        # x = torch.cat([x_tem, x], dim=1)  # c = 256 + 256 = 512
        # x_ptfem = self.ptfem(x)  # 256
        x = torch.cat([x, x_tem, x_lbp1, x_lbp2, x_lbp3], dim=1)
        return x, x_tem


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        """
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        """
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SSPCAB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        '''
            channels: The number of filter at the output (usually the same with the number of filter from the input)
            kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
            dilation: The dilation dimension 'd' from the paper
            reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        '''
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1

        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.se(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.k3 = nn.Sequential(
            nn.Conv2d(256, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(256, planes, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.k1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                      padding=0, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.cov1 = nn.Conv2d(512, 256, 1, 1)

    def forward(self, x):
        identity = self.k1(x)
        N, C, H, W = identity.size()
        out = torch.sigmoid(torch.add(identity, F.adaptive_avg_pool2d(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        x_or = self.k3(identity)
        out = torch.mul(x_or, out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4
        return out
