from __future__ import print_function
import torch

__all__ = ['TrainConfigures']


class TrainConfigures:
    # dataset_path = "/mnt/home/y21301045/datasets"
    dataset_path = r'D:\Datasets\mvtec'
    '''
    数据集位置（mvtec_anomaly_detection.tar.xz 存放位置）
    '''

    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    '''
    模型训练设备
    '''

    batch_size = 2
    '''
    单次训练批次数量
    '''

    worker_num = 0
    '''
    pytorch数据集预加载线程数
    '''

    epoch = 200
    '''
    训练迭代多少次
    '''

    # classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    classes = ['screw', 'carpet', 'capsule', 'metal_nut', 'pill', 'wood', 'transistor', 'cable', 'leather', 'grid', 'tile', 'bottle', 'hazelnut', 'toothbrush', 'zipper']
    '''
    待训练的类别
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
    assert epoch > test_interval, "train epoch must bigger then test interval"

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
