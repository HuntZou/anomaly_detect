import os
import sys
import time

import torch

import train

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..')))

from modules.ad_module import STLNet_AD
from config import TrainConfigures

# TrainConfigures.device = torch.device('cpu')

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
    for i in range(1):
        optimizer = torch.optim.Adam(params=net.parameters())
        input = torch.randn([5, 3, 256, 256]).to(TrainConfigures.device)
        gtmaps = torch.zeros([5, 256, 256]).to(TrainConfigures.device)
        labels = torch.ones([5]).to(TrainConfigures.device)
        start = time.time()
        anorm_heatmap, score_map = net(input)
        optimizer.zero_grad()
        loss = train.fcdd_loss(anorm_heatmap, score_map, gtmaps, labels)
        loss.backward()
        optimizer.step()
        print(f'train done, cost: {time.time() - start}')


if __name__ == '__main__':
    # test_mock()
    train_mock()
