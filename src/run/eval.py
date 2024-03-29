import os
import sys

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..')))

import utils
from config import TrainConfigures
from datasets import load_dataset
from modules.ad_module import STLNet_AD

if __name__ == "__main__":
    for class_name_idx in range(0, len(TrainConfigures.dataset.classes)):
        utils.manual_random_seed()

        class_name = TrainConfigures.dataset.classes[class_name_idx]
        logger.info(f"Start test {class_name}")

        net = STLNet_AD(in_channels=3, pretrained=True, output_stride=16)
        # pre_module_path = TrainConfigures.model_dir(f'{class_name}_LABEL_AUROC')
        pre_module_path = TrainConfigures.model_dir(f'{class_name}_PIXEL_AUROC')

        if not os.path.exists(pre_module_path):
            logger.error(f"{class_name} model not found from path: {pre_module_path}")
            continue
        logger.info(f"load pre-trained {class_name} model from: {pre_module_path}")
        net.load_state_dict(torch.load(pre_module_path, map_location=TrainConfigures.device))

        net = net.to(TrainConfigures.device)

        dataset = 'BTAD'
        supervise_mode = 'malformed_normal_gt'
        preproc = 'lcnaug1'
        noise_mode = 'confetti'
        online_supervision = True
        ds = load_dataset(
            dataset, os.path.abspath(TrainConfigures.dataset.dataset_path), class_name_idx, preproc, supervise_mode,
            noise_mode, online_supervision, TrainConfigures.nominal_label, only_test=True
        )
        test_loader = ds.loaders(train=False)

        net.eval()
        with torch.no_grad():
            test_image_list = list()
            gt_label_list = list()
            gt_mask_list = list()
            score_maps = list()
            for n_batch, data in enumerate(test_loader):
                if n_batch % int(len(test_loader) / 3) == 0:
                    logger.info(f'test {class_name} model,\tprocess: {n_batch} of {len(test_loader)},\tprogress: {int(round(n_batch / len(test_loader), 2) * 100)}%')

                inputs, labels, masks = data
                test_image_list.extend(inputs.cpu().data.numpy())
                gt_label_list.extend(labels.cpu().data.numpy())
                gt_mask_list.extend(masks.cpu().data.numpy())

                # first_img = inputs[0].cpu().detach().permute([1,2,0]).numpy()

                inputs = inputs.to(TrainConfigures.device)

                _, score_map = net(inputs)

                score_maps += torch.norm(score_map, p=2, dim=1).detach().cpu().tolist()

            score_maps = torch.tensor(score_maps, dtype=torch.double).squeeze()
            score_maps = torch.nn.functional.interpolate(score_maps.unsqueeze(1), TrainConfigures.crop_size).squeeze().numpy()
            score_maps = score_maps - score_maps.min()
            score_maps = score_maps / score_maps.max()

            score_labels = np.max(score_maps, axis=(1, 2))
            gt_labels = np.asarray(gt_label_list, dtype=bool)
            label_roc = roc_auc_score(gt_labels, score_labels)
            logger.info(f'{class_name} label_roc: {label_roc}')

            gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
            pixel_roc = roc_auc_score(gt_mask.flatten(), score_maps.flatten())
            logger.info(f'{class_name} pixel_roc: {pixel_roc}')
