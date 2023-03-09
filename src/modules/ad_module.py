import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from modules.stl_net import STL


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=8, backbone='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # resnet 残差之前还有4个层
        self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = torch.cat([
            x[:, :, 0: int(x.shape[-2] / 2), 0: int(x.shape[-1] / 2)],
            x[:, :, int(x.shape[-2] / 2):, 0: int(x.shape[-1] / 2)],
            x[:, :, 0: int(x.shape[-2] / 2), int(x.shape[-1] / 2):],
            x[:, :, int(x.shape[-2] / 2):, int(x.shape[-1] / 2):],
        ], dim=1)
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features_1 = x
        x = self.layer2(x)
        low_level_features_2 = x
        x = self.layer3(x)
        low_level_features_3 = x

        return x, low_level_features_1, low_level_features_2, low_level_features_3

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False


# ASPP
def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channles),
    )


def conv3(in_channels, out_channles):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channles)
    )


class SSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSP, self).__init__()
        self.spp1 = conv3(in_channels, out_channels)
        self.spp2 = nn.Sequential(
            conv3(in_channels, out_channels),
            conv3(out_channels, out_channels)
        )
        self.spp3 = nn.Sequential(
            conv3(in_channels, out_channels),
            conv3(out_channels, out_channels),
            conv3(out_channels, out_channels)
        )

        self.cov = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x1 = self.spp1(x)
        x2 = self.spp2(x)
        x3 = self.spp3(x)
        x = self.cov(torch.cat((x1, x2, x3), dim=1))
        return x


class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 4, 6, 12]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(256 * 5, 256, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.relu(x)
        x = self.dropout(self.bn1(x))

        return x


def feature_cat(low_level_features_1, low_level_features_2):
    H, W = low_level_features_1.size(2), low_level_features_1.size(3)
    x = F.interpolate(low_level_features_2, size=(H, W), mode='bilinear', align_corners=True)
    x = torch.cat((low_level_features_1, x), dim=1)
    return x


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


class STLNet_AD(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, output_stride=8):
        super(STLNet_AD, self).__init__()
        self.backbone = ResNet(in_channels=in_channels, output_stride=16, pretrained=pretrained)
        self.STL = STL(in_channel=768)  # 192  768 512
        self.self_calibration = SelfCalibration(1024, 256, stride=1, padding=1, dilation=1, groups=1, pooling_r=4)
        self.conv2 = nn.Conv2d(1216, 128, 3, 1, 1)  # 1280
        self.conv_final = nn.Conv2d(128, 32, 1)
        self.conv_score = nn.Conv2d(32, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        # ********原始文章中接在resnet后的模块
        self.ASSP = ASSP(in_channels=1024, output_stride=16)
        self.DenseASPP = DenseASPP(in_channels=1024)
        # ********

    def forward(self, x):
        x, low_level_features_1, low_level_features_2, low_level_features_3 = self.backbone(x)
        low_level_features = self.STL(self.feature_cat(low_level_features_1, low_level_features_2))
        low_level_features_3 = self.self_calibration(low_level_features_3)
        low_level_features = self.feature_cat(low_level_features, low_level_features_3)
        low_level_features = self.conv2(low_level_features)

        anorm_heatmap = self.conv_final(low_level_features)
        score_map = self.conv_score(anorm_heatmap)

        return anorm_heatmap, score_map

    def feature_cat(self, low_level_features_1, low_level_features_2):
        H, W = low_level_features_1.size(2), low_level_features_1.size(3)
        x = F.interpolate(low_level_features_2, size=(H, W), mode='bilinear', align_corners=True)
        # x = self.layer2_cov(x)
        x = torch.cat((low_level_features_1, x), dim=1)
        return x


class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """

    def __init__(self, in_channels):
        super(DenseASPP, self).__init__()

        dropout0 = 0.1
        d_feature0 = 256
        d_feature1 = 128
        num_features = 256

        self.conv_2 = nn.Conv2d(in_channels, 256, 1, 1, 1)
        self.conv_3 = nn.Conv2d(896, 128, 1, 1, 1)

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # initialize_weights(self)

    def forward(self, _input):
        feature = self.conv_2(_input)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        feature = self.conv_3(feature)

        return feature


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.drop_rate = drop_out

        if bn_start:
            self.dense = nn.Sequential(
                nn.BatchNorm2d(input_num, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate),
                nn.Dropout(drop_out)

            )
        else:
            self.dense = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate),
                nn.Dropout(drop_out)

            )

    def forward(self, _input):
        # feature = super(_DenseAsppBlock, self).forward(_input)
        feature = self.dense(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class SelfCalibration(nn.Module):
    """
    自校准模块
    """

    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(SelfCalibration, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(inplanes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(inplanes),
        )

        self.k1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                      padding=0, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        x_or = self.k3(x)
        out = torch.mul(x_or, out)  # k3 * sigmoid(identity + k2)
        out = self.k1(out)  # k4
        return out
