import pathlib

import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

from modules.ad_module import STLNet_AD
from config import get_args
from datasets import load_dataset
from visualize import *


def main(c):
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                   'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    for class_name in range(0, 5):
        c.class_name = CLASS_NAMES[class_name]
        print('class_name:', c.class_name)

        c.img_size = (c.input_size, c.input_size)  # HxW format
        c.crp_size = (c.input_size, c.input_size)  # HxW format
        c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        c.img_dims = [3] + list(c.img_size)
        c.model = 'mvtec'

        # datadir = '/media/ahu/ac724e7d-d8c2-45a4-9df8-304a70f0335f/ZUO/anomaly_detection/data'
        dataset_path = c.dataset_path
        dataset = 'mvtec'
        supervise_mode = 'malformed_normal_gt'
        preproc = 'lcnaug1'
        noise_mode = 'confetti'
        online_supervision = True
        nominal_label = c.nominal_label

        ds = load_dataset(
            dataset, os.path.abspath(dataset_path), class_name, preproc, supervise_mode,
            noise_mode, online_supervision, nominal_label,
        )  # 返回一个类

        test_loader = ds.loaders(batch_size=c.batch_size, num_workers=c.workers, train=False)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        c.use_cuda = not c.no_cuda and torch.cuda.is_available()
        c.device = torch.device("cuda" if c.use_cuda else "cpu")

        det_roc_obs = Score_Observer('DET_AUROC')
        seg_roc_obs = Score_Observer('SEG_AUROC')
        seg_pro_obs = Score_Observer('SEG_AUPRO')

        for epoch in range(0, 201):
            net = STLNet_AD(in_channels=3, pretrained=True, output_stride=16)
            model_dir = os.path.join('model', c.class_name)
            dir = os.path.join(model_dir, 'STL_{:d}.pt'.format(epoch))
            checkpoint = torch.load(dir)
            os.remove(dir)
            net.load_state_dict(checkpoint['net'])
            net = net.to(c.device)
            # optimizer = optim.SGD(net.parameters(), lr=c.lr, weight_decay=c.weight_decay, momentum=0.9, nesterov=True)
            # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: c.sched_param[0] ** ep)

            test_image_list = list()
            gt_label_list = list()
            gt_mask_list = list()
            test_list = list()
            net = net.eval()
            with torch.no_grad():
                for n_batch, data in enumerate(test_loader):
                    inputs, labels, masks = data
                    test_image_list.extend(t2np(inputs))
                    gt_label_list.extend(t2np(labels))
                    gt_mask_list.extend(t2np(masks))
                    # inputs = inputs.cuda(device=device_ids[0])
                    # masks = masks.cuda(device=device_ids[0])
                    inputs = inputs.to(c.device)
                    masks = masks.to(c.device)
                    outputs, outputs1, _ = net(inputs)
                    score = score2(outputs1)
                    test_list = test_list + score.detach().cpu().tolist()
                    # print('Epoch:{:d} \t test_loss: {:.4f}'.format(epoch, score.mean()))
            torch.cuda.empty_cache()
            print('Epoch: {:d}'.format(epoch))

            test_norm = torch.tensor(test_list, dtype=torch.double).squeeze()
            test_norm = F.interpolate(test_norm.unsqueeze(1), c.crp_size).squeeze().numpy()

            super_mask = test_norm
            super_mask = super_mask - super_mask.min()
            super_mask = super_mask / super_mask.max()
            # super_mask = super_mask / np.percentile(super_mask, 99)
            # super_mask[super_mask > 1.0] = 1.0

            score_label = np.max(super_mask, axis=(1, 2))
            gt_label = np.asarray(gt_label_list, dtype=bool)
            det_roc_auc = roc_auc_score(gt_label, score_label)
            _ = det_roc_obs.update(100.0 * det_roc_auc, epoch)
            # calculate segmentation AUROC
            gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
            seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
            save_best_seg_weights = seg_roc_obs.update(100.0 * seg_roc_auc, epoch)
            # if save_best_seg_weights and c.action_type != 'norm-test':
            #     save_weights(net, c.model, run_date)  # avoid unnecessary saves
            # calculate segmentation AUPRO
            # from https://github.com/YoungGod/DFR:
            if c.pro:  # and (epoch % 4 == 0):  # AUPRO is expensive to compute
                max_step = 1000
                expect_fpr = 0.3  # default 30%
                max_th = super_mask.max()
                min_th = super_mask.min()
                delta = (max_th - min_th) / max_step
                ious_mean = []
                ious_std = []
                pros_mean = []
                pros_std = []
                threds = []
                fprs = []
                binary_score_maps = np.zeros_like(super_mask, dtype=bool)
                for step in range(max_step):
                    thred = max_th - step * delta
                    # segmentation
                    binary_score_maps[super_mask <= thred] = 0
                    binary_score_maps[super_mask > thred] = 1
                    pro = []  # per region overlap
                    iou = []  # per image iou
                    # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
                    # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
                    for i in range(len(binary_score_maps)):  # for i th image
                        # pro (per region level)
                        label_map = label(gt_mask[i], connectivity=2)
                        props = regionprops(label_map)
                        for prop in props:
                            x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                            cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                            # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                            cropped_mask = prop.filled_image  # corrected!
                            intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                            pro.append(intersection / prop.area)
                        # iou (per image level)
                        intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                        union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                        if gt_mask[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                            iou.append(intersection / union)
                    # against steps and average metrics on the testing data
                    ious_mean.append(np.array(iou).mean())
                    # print("per image mean iou:", np.array(iou).mean())
                    ious_std.append(np.array(iou).std())
                    pros_mean.append(np.array(pro).mean())
                    pros_std.append(np.array(pro).std())
                    # fpr for pro-auc
                    gt_masks_neg = ~gt_mask
                    fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
                    fprs.append(fpr)
                    threds.append(thred)
                    # as array
                threds = np.array(threds)
                pros_mean = np.array(pros_mean)
                pros_std = np.array(pros_std)
                fprs = np.array(fprs)
                ious_mean = np.array(ious_mean)
                ious_std = np.array(ious_std)
                # best per image iou
                best_miou = ious_mean.max()
                # print(f"Best IOU: {best_miou:.4f}")
                # default 30% fpr vs pro, pro_auc
                idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
                fprs_selected = fprs[idx]
                fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
                pros_mean_selected = pros_mean[idx]
                seg_pro_auc = auc(fprs_selected, pros_mean_selected)
                _ = seg_pro_obs.update(100.0 * seg_pro_auc, epoch)
                #

                # save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)
                # export visualuzations
            break
        if c.viz:
            precision, recall, thresholds = precision_recall_curve(gt_label, score_label)
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            det_threshold = thresholds[np.argmax(f1)]
            print('Optimal DET Threshold: {:.2f}'.format(det_threshold))
            precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), super_mask.flatten())
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            seg_threshold = thresholds[np.argmax(f1)]
            print('Optimal SEG Threshold: {:.2f}'.format(seg_threshold))
            export_groundtruth(c, test_image_list, gt_mask)
            export_scores(c, test_image_list, super_mask, seg_threshold)
            export_test_images(c, test_image_list, gt_mask, super_mask, seg_threshold)
            export_hist(c, gt_mask, super_mask, seg_threshold)


def score2(outs):
    outs = torch.norm(outs, p=2, dim=1)
    # outs = abs(outs)
    outs = outs.squeeze()
    return outs


def score1(outs):
    outs = torch.norm(outs, p=2, dim=1)
    loss = outs ** 2  # 张量间基本计算
    # loss = 1-(-((loss +1).sqrt() - 1)).exp()
    loss = (loss + 1).sqrt() - 1
    # loss = loss.sqrt()
    # loss = F.interpolate(loss, size=c.crp_size, mode='bilinear', align_corners=True)
    # loss = 1-(-loss).exp()
    loss = loss.squeeze()
    return loss


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.epoch_last = epoch
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score()

        return save_weights

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score, self.max_epoch))
        result_dir = os.path.join('../output/result', c.class_name)
        pathlib.Path(os.path.split(result_dir)[0]).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(result_dir, 'result_test.txt'), 'a') as file:  # 在输出文件夹打开或创建results_train.txt文件
            file.write('epoch_last: {:d} \n'.format(self.epoch_last))
            file.write('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d} \n'.format(
                self.name, self.last, self.max_score, self.max_epoch))  # 往打开的文件里写入批次等信息


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


if __name__ == '__main__':
    c = get_args()
    main(c)
