import os
import sys

import matplotlib.pyplot as plt
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..')))

import utils
from config import TrainConfigures
from datasets.mvtec import ADMvTec

generation_count = 300

if __name__ == '__main__':
    supervise_mode = 'malformed_normal_gt'
    preproc = 'lcnaug1'
    noise_mode = 'confetti'
    online_supervision = True

    for class_name_idx in range(0, len(TrainConfigures.classes)):
        class_name = TrainConfigures.classes[class_name_idx]
        dataset = ADMvTec(
            root=os.path.abspath(TrainConfigures.dataset_path), normal_class=class_name_idx, preproc=preproc,
            supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
            nominal_label=TrainConfigures.nominal_label, only_test=False
        )

        for i in range(generation_count):
            pseudo, label, ground_truth = dataset.train_set[i % len(dataset.train_set)]
            if i % int(generation_count / 10) == 0:
                logger.info(f'test {class_name} model,\ttotal len: {len(dataset.train_set)},\tprocess: {i} of {generation_count},\tprogress: {int(round(i / generation_count, 2) * 100)}%')

            fig_img, ax_img = plt.subplots(1, 2)
            ax_img[0].set_title('pseudo image')
            ax_img[0].imshow(((pseudo - pseudo.min()) / (pseudo.max() - pseudo.min())).cpu().detach().permute([1, 2, 0]))
            ax_img[1].set_title('ground truth')
            ax_img[1].imshow(ground_truth.cpu().detach().squeeze(0))
            image_file = os.path.join(utils.get_dir(TrainConfigures.output_dir, "pseudo", class_name), f"{i}_{'anon' if label == 1 else 'norm'}.png")
            # image_file = os.path.join(utils.get_dir(r"D:\Tmp\output", "pseudo", class_name), f"{i}_{'anon' if label == 1 else 'norm'}.png")
            fig_img.savefig(image_file, format='png')
            plt.close()
