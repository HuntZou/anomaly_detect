from __future__ import print_function

import os
import platform

import torch

import utils

__all__ = ['TrainConfigures']


class TrainConfigures:
    # 默认在windows下debug，linux下真正训练。
    # dataset = utils.BTAD(debug='Windows' == platform.system())
    dataset = utils.MVTec(debug='Windows' == platform.system())

    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    '''
    模型训练设备
    '''

    epoch = 200
    '''
    训练迭代多少次
    '''

    visualize = False
    '''
    是否可视化结果
    '''

    crop_size = (256, 256)
    '''
    训练过程中要再次将图片resize，太大了就会内存不足
    '''

    learn_rate = 1e-3
    '''
    学习率
    '''

    test_interval = 20
    '''
    每训练多少个epoch后使用测试数据集测试模型
    '''

    span_of_best = 4
    """
    多少个 test_interval * epoch 没有得到新的最优值后，认为之前已经得到最优值了
    """

    assert epoch > test_interval, "train epoch must greater then test interval"

    nominal_label = 0
    '''
    Determines the label that marks nominal samples.
    Note that this is not the class that is considered nominal!
    For instance, class 5 is the nominal class, which is labeled with the nominal label 0.
    '''

    weight_decay = 3e-5
    '''
    SGD的一个参数
    '''

    sched_param = 0.995
    '''
    优化器的一个参数，指定学习率在每轮训练后的调整因数
    '''

    calc_aupro = False
    '''
    是否在一次训练完成后计算模型的aupro
    '''

    project_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))
    """
    项目的根目录路径
    """

    output_dir = os.path.join(project_dir, "output")
    """
    输出文件目录
    """

    model_dir = lambda class_name: os.path.join(utils.get_dir(TrainConfigures.output_dir, "modules"), f'{class_name}_{"_".join([str(i) for i in TrainConfigures.crop_size])}.pth')
    """
    模型输出目录
    """

    export_result_img_dir = lambda class_name: os.path.join(TrainConfigures.output_dir, "visualize", "mvtec", class_name)
    """
    模型处理测试图片后的导出路径
    """
