import datetime, time
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
from loguru import logger

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1 / 2.54
dpi = 300
denormalization_export_norm_mean, denormalization_export_norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def export_test_images(test_img, gts, scores, threshold, export_dir):
    if not os.path.isdir(export_dir):
        logger.info(f'start export {len(test_img)} test images to: {os.path.abspath(export_dir)}')
        export_start_time = time.time()
        os.makedirs(export_dir, exist_ok=True)
        num = len(test_img)
        scores_norm = 1.0 / scores.max()
        for i in range(num):
            img = test_img[i]
            img = (img.transpose(1, 2, 0) - np.min(img)) / (np.max(img) - np.min(img))
            # gts
            gt_mask = gts[i].astype(np.float64)
            # gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0 * gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] > threshold] = 1.0
            # score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0 * score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0 * scores[i] * scores_norm).astype(np.uint8)
            fig_img, ax_img = plt.subplots(2, 2)
            ax_img[0][0].set_title('original image')
            ax_img[0][0].imshow(img)
            ax_img[0][1].set_title('ground truth')
            ax_img[0][1].imshow(gt_img)
            ax_img[1][0].set_title('score map')
            ax_img[1][0].imshow(score_map, cmap='jet', norm=norm)
            ax_img[1][1].set_title('prediction')
            ax_img[1][1].imshow(score_img)
            image_file = os.path.join(export_dir, '{:08d}.jpg'.format(i))
            fig_img.savefig(image_file, format='jpg')
            plt.close()
        logger.info(f'Export test image done, cost time: {time.time() - export_start_time}')
