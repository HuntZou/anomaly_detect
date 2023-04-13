import os
import sys
import time

import torch

import train

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..')))

from modules.ad_module import STLNet_AD
from config import TrainConfigures
from datasets.mvtec import ADMvTec

TrainConfigures.device = torch.device('cuda:0')

net = STLNet_AD(in_channels=3, pretrained=True, output_stride=16).to(TrainConfigures.device)


def test_mock():
    input = torch.randn([5, 3, 256, 256]).to(TrainConfigures.device)
    net.eval()
    with torch.no_grad():
        for i in range(1):
            start = time.time()
            _, score_map = net(input)
            print(f'test done, cost: {time.time() - start}')


def train_mock():
    optimizer = torch.optim.Adam(params=net.parameters())
    batch_size = 5
    for i in range(15):
        inputs = torch.randn([batch_size, 3, 256, 256]).to(TrainConfigures.device)
        gtmaps = torch.zeros([batch_size, 256, 256]).to(TrainConfigures.device)
        labels = torch.ones([batch_size]).to(TrainConfigures.device)
        start = time.time()
        anorm_heatmap, score_map = net(inputs)
        optimizer.zero_grad()
        loss = train.fcdd_loss(anorm_heatmap, score_map, gtmaps, labels)
        loss.backward()
        optimizer.step()
        print(f'{time.ctime()} epoch {i}, train cost: {time.time() - start}')


def train_real():
    class_name_idx = 1
    batch_size = 2
    class_name = TrainConfigures.dataset.classes[class_name_idx]
    print(f'class_name: {class_name}')
    dataset = ADMvTec(
        root=os.path.abspath(TrainConfigures.dataset.dataset_path), normal_class=class_name_idx, preproc='lcnaug1',
        supervise_mode='malformed_normal_gt', noise_mode='confetti', online_supervision=True,
        nominal_label=TrainConfigures.nominal_label, only_test=False
    )
    train_loader = torch.utils.data.DataLoader(dataset=dataset.train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=False,
                                               prefetch_factor=2, persistent_workers=TrainConfigures.dataset.worker_num > 0)

    optimizer = torch.optim.Adam(params=net.parameters())
    for n_batch, data in enumerate(train_loader):
        inputs, labels, gtmaps = data[0].to(TrainConfigures.device, non_blocking=True), data[1], data[2].to(TrainConfigures.device, non_blocking=True)
        start = time.time()
        anorm_heatmap, score_map = net(inputs)
        optimizer.zero_grad()
        loss = train.fcdd_loss(anorm_heatmap, score_map, gtmaps, labels)
        loss.backward()
        optimizer.step()
        print(f'{time.ctime()} iter {n_batch}, train cost: {time.time() - start}')


if __name__ == '__main__':
    # test_mock()
    train_mock()
    # train_real()
