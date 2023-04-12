import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..')))

from modules.ad_module import STLNet_AD
from config import TrainConfigures

TrainConfigures.device = torch.device('cpu')

net = STLNet_AD(in_channels=3, pretrained=True, output_stride=16).to(TrainConfigures.device)
input = torch.randn([1, 3, 256, 256]).to(TrainConfigures.device)
net.eval()
with torch.no_grad():
    for i in range(10):
        start = time.time()
        _, score_map = net(input)
        print(f'cost: {time.time() - start}')
