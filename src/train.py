import itertools
import os
from os.path import join as path_join

import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import mahalanobis
from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

from loguru import logger
from modules.ad_module import STLNet_AD
from config import TrainConfigures
from datasets import load_dataset
import numpy as np
import visualize
import utils

project_dir = os.path.abspath("../")


def main():
    for class_name_idx in range(0, len(TrainConfigures.classes)):
        class_name = TrainConfigures.classes[class_name_idx]
        board = SummaryWriter(path_join(project_dir, "output", "logs", class_name))
        logger.info(f'start training class: {class_name}')

        dataset = 'BTAD'
        supervise_mode = 'malformed_normal_gt'
        preproc = 'lcnaug1'
        noise_mode = 'confetti'
        online_supervision = True

        ds = load_dataset(
            dataset, os.path.abspath(TrainConfigures.dataset_path), class_name_idx, preproc, supervise_mode,
            noise_mode, online_supervision, TrainConfigures.nominal_label,
        )

        train_loader, test_loader = ds.loaders(batch_size=TrainConfigures.batch_size, num_workers=TrainConfigures.worker_num)

        net = STLNet_AD(in_channels=3, pretrained=True, output_stride=16)

        # load pre-train module if exist
        pre_module_path = os.path.join(utils.get_dir(project_dir, "output", "modules"), f'{class_name}_{"_".join([str(i) for i in TrainConfigures.crop_size])}.pth')
        if os.path.exists(pre_module_path):
            logger.info(f'load pre-train module: {class_name}')
            net.load_state_dict(torch.load(pre_module_path, map_location=TrainConfigures.device))

        net = net.to(TrainConfigures.device)
        optimizer = optim.SGD(net.parameters(), lr=TrainConfigures.learn_rate, weight_decay=TrainConfigures.weight_decay, momentum=0.9, nesterov=True)
        lr_optimizer = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: TrainConfigures.sched_param ** ep, verbose=True)

        label_roc_observer = ScoreObserver('LABEL_AUROC', class_name, TrainConfigures.epoch, TrainConfigures.epoch * 3, threshold=4)
        pixel_roc_observer = ScoreObserver('PIXEL_AUROC', class_name, TrainConfigures.epoch, TrainConfigures.epoch * 3, threshold=4)
        pixel_pro_observer = ScoreObserver('PIXEL_AUPRO', class_name, TrainConfigures.epoch, TrainConfigures.epoch * 3, threshold=4)

        for epoch in itertools.count():

            net = net.train()
            loss_mean = 0
            for n_batch, data in enumerate(train_loader):
                if n_batch % int(len(train_loader) / 10) == 0:
                    logger.info(f'epoch: {epoch}, \t class: {class_name}, \t process: {n_batch} of {len(train_loader)}, \t progress: {round(n_batch / len(train_loader), 2) * 100}%')
                inputs, labels, gtmaps = data[0].to(TrainConfigures.device, non_blocking=True), data[1], data[2].to(TrainConfigures.device, non_blocking=True)
                anorm_heatmap, score_map = net(inputs)
                optimizer.zero_grad()
                loss = fcdd_loss(anorm_heatmap, score_map, gtmaps, labels)
                loss.backward()
                optimizer.step()
                loss_mean += loss.item() / len(train_loader)
            lr_optimizer.step()

            board.add_scalar(f"loss/train_loss_mean", loss_mean, epoch)
            total, non_zero = count_non_zeros(net)
            board.add_scalar(f"variable/non_zero_ratio", round(non_zero / total, 2), epoch)
            board.add_scalar(f"variable/non_zero_count", non_zero, epoch)

            if epoch % TrainConfigures.test_interval == 0:
                net = net.eval()
                test_image_list = list()
                gt_label_list = list()
                gt_mask_list = list()
                score_maps = list()
                with torch.no_grad():
                    logger.info(f'start testing, test dataset length: {len(test_loader)}')
                    for n_batch, data in enumerate(test_loader):
                        inputs, labels, masks = data
                        test_image_list.extend(t2np(inputs))
                        gt_label_list.extend(t2np(labels))
                        gt_mask_list.extend(t2np(masks))
                        inputs = inputs.to(TrainConfigures.device)
                        _, score_map = net(inputs)

                        score_maps += score2(score_map).detach().cpu().tolist()

                score_maps = torch.tensor(score_maps, dtype=torch.double).squeeze()
                score_maps = F.interpolate(score_maps.unsqueeze(1), TrainConfigures.crop_size).squeeze().numpy()
                score_maps = score_maps - score_maps.min()
                score_maps = score_maps / score_maps.max()

                score_labels = np.max(score_maps, axis=(1, 2))
                gt_labels = np.asarray(gt_label_list, dtype=bool)
                label_roc = roc_auc_score(gt_labels, score_labels)
                best_label_roc_already = label_roc_observer.update(100.0 * label_roc, epoch, net)
                board.add_scalar("ROC/label_roc", label_roc, epoch)

                gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
                pixel_roc = roc_auc_score(gt_mask.flatten(), score_maps.flatten())
                best_pixel_roc_already = pixel_roc_observer.update(100.0 * pixel_roc, epoch, net)
                board.add_scalar("ROC/pixel_roc", pixel_roc, epoch)

                if TrainConfigures.calc_aupro:
                    """
                    calculate segmentation AUPRO
                    AUPRO is expensive to compute
                    from https://github.com/YoungGod/DFR
                    """
                    max_step = 1000
                    expect_fpr = 0.3  # default 30%
                    max_th = score_maps.max()
                    min_th = score_maps.min()
                    delta = (max_th - min_th) / max_step
                    ious_mean = []
                    ious_std = []
                    pros_mean = []
                    pros_std = []
                    threds = []
                    fprs = []
                    binary_score_maps = np.zeros_like(score_maps, dtype=bool)
                    for step in range(max_step):
                        thred = max_th - step * delta
                        # segmentation
                        binary_score_maps[score_maps <= thred] = 0
                        binary_score_maps[score_maps > thred] = 1
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
                    _ = pixel_pro_observer.update(100.0 * seg_pro_auc, epoch, net)

                # got best result
                if best_label_roc_already and best_pixel_roc_already:
                    board.add_graph(net, torch.randn_like(inputs, device=TrainConfigures.device))
                    logger.info(f'class: {class_name} train done at epoch: {epoch}, \t'
                                f'best_label_roc: {round(label_roc_observer.max_score, 2)} at epoch: {round(label_roc_observer.max_epoch, 2)}, \t'
                                f'best_pixel_roc: {round(pixel_roc_observer.max_score, 2)} at epoch: {round(pixel_roc_observer.max_epoch, 2)}')
                    if TrainConfigures.visualize:
                        precision, recall, thresholds = precision_recall_curve(gt_labels, score_labels)
                        a = 2 * precision * recall
                        b = precision + recall
                        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                        det_threshold = thresholds[np.argmax(f1)]
                        logger.info('Optimal LABEL Threshold: {:.2f}'.format(det_threshold))
                        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score_maps.flatten())
                        a = 2 * precision * recall
                        b = precision + recall
                        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                        seg_threshold = thresholds[np.argmax(f1)]
                        logger.info('Optimal PIXEL Threshold: {:.2f}'.format(seg_threshold))
                        visualize.export_test_images(class_name, test_image_list, gt_mask, score_maps, seg_threshold)
                    break


def fcdd_loss(anorm_heatmap, score_map, gtmaps, labels):
    # loss = torch.norm(outs, p=2, dim=1).unsqueeze(1)
    # loss = loss ** 2  # 张量间基本计算
    # loss = (loss + 1).sqrt() - 1
    loss1 = __fcdd_pixloss(anorm_heatmap, gtmaps)
    loss2 = __f_score_sim(anorm_heatmap, score_map, labels)
    loss3 = __score_loss(score_map, gtmaps)
    # loss_tem = __fcdd_pixloss(x_tem, gtmaps)
    # loss_tem = se_loss(x_tem, gtmaps)
    # loss_ad = __adjacent_loss(outs, labels)
    # print(loss1, 10*loss2, loss3)
    loss = loss1 + loss3 + loss2
    return loss


def __fcdd_loss(outs, outs1, gtmaps, labels):
    # loss_sim = __pixel_sim_loss(outs1, labels)
    loss_ad = __adjacent_loss(outs, labels)
    outs = torch.norm(outs, p=2, dim=1).unsqueeze(1)
    loss = outs ** 2  # 张量间基本计算
    loss = (loss + 1).sqrt() - 1
    loss = __fcdd_pixloss(loss, gtmaps)
    # loss = loss + loss_sim + loss_ad
    loss = loss + 0.5 * loss_ad

    return loss


def __Gau_loss(outs, gtmaps, ref_mean, ref_var):
    outs = (outs - ref_mean) / ref_var
    loss = __fcdd_pixloss(outs, gtmaps)
    return abs(loss)


def __fcdd_pixloss(anorm_heatmap, gtmaps):
    anorm_heatmap = torch.norm(anorm_heatmap, p=2, dim=1).unsqueeze(1)
    anorm_heatmap = anorm_heatmap ** 2  # 张量间基本计算
    anorm_heatmap = (anorm_heatmap + 1).sqrt() - 1
    anorm_heatmap = F.interpolate(anorm_heatmap, size=TrainConfigures.crop_size, mode='bilinear', align_corners=True)
    N, C, H, W = anorm_heatmap.size()
    P = N * C * H * W
    anorm_heatmap = anorm_heatmap.reshape(P)
    gtmaps = gtmaps.reshape(P)
    # loss_ = __norm_anom_margin(loss, gtmaps)
    norm = anorm_heatmap[gtmaps == 0]  # loss 张量对应于label张量元素为0的相应位置，形成的张量赋值给norma
    # norm = loss[gtmaps==0]-2.5
    # norm[norm<0]=0
    if 1 in gtmaps:
        anom = (-(((1 - (-anorm_heatmap[gtmaps == 1]).exp()) + 1e-31).log()))  # 防止为0
        # anom = (-loss[gtmaps == 1]).exp()
        # print(norm.mean(), loss[gtmaps == 1].mean())
        anorm_heatmap = 0.5 * norm.mean() + 0.5 * anom.mean()

    else:
        anom = 0
        anorm_heatmap = norm.mean()

    return anorm_heatmap


def margin_mean(loss, gtmaps):
    loss = torch.norm(loss, p=2, dim=1).unsqueeze(1)
    # loss = loss ** 2  # 张量间基本计算
    # loss = (loss + 1).sqrt() - 1
    loss = F.interpolate(loss, size=TrainConfigures.crop_size, mode='bilinear', align_corners=True)
    N, C, H, W = loss.size()
    P = N * C * H * W
    loss = loss.reshape(P)
    gtmaps = gtmaps.reshape(P)
    norm = abs(10 - loss[gtmaps == 0]).mean()
    anom = 10 - loss[gtmaps == 1]
    anom[anom < 0] = 0
    # norm[norm < 0] = 0
    # loss = abs(10-anom).mean() + norm
    # loss = abs(10-loss).mean()
    loss = 0.5 * anom.mean() + 0.5 * norm
    # print(0.01*abs(100-anom), norm)
    return loss


def __supervised_loss(loss, labels):
    loss = loss.reshape(labels.size(0), -1).mean(-1)
    norm = loss[labels == 0]  # loss 张量对应于label张量元素为0的相应位置，形成的张量赋值给normal
    anom = (-(((1 - (-loss[labels == 1]).exp()) + 1e-31).log()))  # 防止为0
    loss[(1 - labels).nonzero().squeeze()] = norm
    loss[labels.nonzero().squeeze()] = anom
    return loss.mean()


def __gt_loss(loss, gtmaps):
    loss = F.interpolate(loss, size=TrainConfigures.crop_size, mode='bilinear', align_corners=True)
    norm = (loss * (1 - gtmaps)).view(loss.size(0), -1).mean(-1)
    exclude_complete_nominal_samples = ((gtmaps == 1).view(gtmaps.size(0), -1).sum(-1) > 0)
    anom = torch.zeros_like(norm)
    if exclude_complete_nominal_samples.sum() > 0:
        a = (loss * gtmaps)[exclude_complete_nominal_samples]
        anom[exclude_complete_nominal_samples] = (
            -(((1 - (-a.view(a.size(0), -1).mean(-1)).exp()) + 1e-31).log())
        )
        # anom[exclude_complete_nominal_samples] = (
        #     -(((1 - (-a.view(a.size(0), -1)).exp()) + 1e-10).log())
        # ).mean(-1)
    loss = norm + anom

    return loss


def __adjacent_loss(outs, labels):
    norm = outs[labels == 0]  # loss 张量对应于label张量元素为0的相应位置，形成的张量赋值给normal
    N, C, H, W = norm.size()
    B = int(N / 2)
    if N % 2 == 0:
        feat_p = norm[:B, :, :, :]
        feat_f = norm[B:, :, :, :]
    else:
        feat_p = norm[:B, :, :, :]
        feat_f = norm[B: -1, :, :, :]

    feat_p = F.normalize(feat_p, p=2, dim=1)
    feat_f = F.normalize(feat_f, p=2, dim=1)
    sim_dis = torch.tensor(0.).to(TrainConfigures.device)
    for i in range(B):
        s_sim_map = pair_wise_sim_map(feat_p[i], feat_p[i])
        t_sim_map = pair_wise_sim_map(feat_f[i], feat_f[i])
        p_s = F.log_softmax(s_sim_map / 1.0 + 1e-31, dim=1)
        p_t = F.softmax(t_sim_map / 1.0, dim=1)

        sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
        sim_dis += sim_dis_
    sim_dis = sim_dis / B
    return sim_dis


def __f_score_sim(anorm_heatmap, score_map, labels):
    anorm_heatmap = anorm_heatmap[labels == 0]
    score_map = score_map[labels == 0]
    N, C, H, W = anorm_heatmap.size()
    sim_dis = torch.tensor(0.).to(TrainConfigures.device)
    # outs = F.normalize(outs, p=2, dim=1)
    # outs1 = F.normalize(outs1, p=2, dim=1)
    for i in range(N):
        # s_map = pair_wise_sim_map(outs1[i], outs1[i])
        # f_map = pair_wise_sim_map(outs[i], outs[i])
        s_map = sim_distance(score_map[i], 2, 4)
        # s_map = pair_distance(outs[i])
        f_map = sim_distance(anorm_heatmap[i], 2, 4)
        # p_f = F.log_softmax(f_map / 1.0 + 1e-31, dim=1)
        p_s = F.log_softmax(s_map / 1.0, dim=1)
        p_f = F.softmax(f_map / 1.0, dim=1)
        # sim_dis_ = F.kl_div(p_s, p_f, reduction='batchmean')
        # sim_dis += sim_dis_
        sim_dis += F.kl_div(p_s, p_f, reduction='batchmean')
    sim_dis = (sim_dis / N) if N != 0 else 1

    # s_map = pair_distance(outs1)
    # f_map = pair_distance(outs)
    # p_s = F.log_softmax(s_map / 1.0 + 1e-31, dim=1)
    # p_f = F.softmax(f_map / 1.0, dim=1)
    # sim_dis = F.kl_div(p_s, p_f, reduction='batchmean')

    return sim_dis


def __f_score_sim1(outs, outs1, labels):
    outs = outs[labels == 0]
    outs1 = outs1[labels == 0]
    N, C, H, W = outs.size()
    P = H * W
    sim_dis = torch.tensor(0.).to(TrainConfigures.device)
    a_j = torch.tensor([0, 1])
    a_j = a_j.repeat(int(P / 2))
    # outs = F.normalize(outs, p=2, dim=1)
    for i in range(N):
        s_ = outs1[i].reshape(1, P).permute(1, 0)
        f_ = outs[i].reshape(C, P).permute(1, 0)
        # P = int(P/2)*2
        # s_ = s_[:P, :]
        # f_ = f_[:P, :]
        # s_ = s_[a_j==0]
        # f_ = f_[a_j==0]
        perm1 = torch.randperm(P)
        perm2 = torch.randperm(P)
        s_1 = s_[perm1[:1024], :]
        f_1 = f_[perm1[:1024], :]
        s_2 = s_[perm2[:1024], :]
        f_2 = f_[perm2[:1024], :]

        s_map = pair_distance1(s_1, s_2)
        f_map = pair_distance1(f_1, f_2)
        p_s = F.log_softmax(s_map / 1.0, dim=1)
        p_f = F.softmax(f_map / 1.0, dim=1)
        # p_f = F.log_softmax(f_map / 1.0, dim=1)
        # p_s = F.softmax(s_map / 1.0, dim=1)

        sim_dis += F.kl_div(p_s, p_f, reduction='batchmean')
    sim_dis = sim_dis / N

    return sim_dis


def __score_loss(score_map, gtmaps):
    # loss = abs(loss)
    # loss = torch.sigmoid(loss)
    score_map = torch.norm(score_map, p=2, dim=1).unsqueeze(1)
    score_map = F.interpolate(score_map, size=TrainConfigures.crop_size, mode='bilinear', align_corners=True)
    N, C, H, W = score_map.size()
    P = N * C * H * W
    score_map = score_map.reshape(P)
    gtmaps = gtmaps.reshape(P)
    loss_ = __norm_anom_loss(score_map, gtmaps)
    norm = score_map[gtmaps == 0]
    # anom = loss[gtmaps == 1]
    norm = norm - 2.5
    norm[norm < 0] = 0
    # anom = 10-loss[gtmaps == 1]
    # anom[anom < 0] = 0
    score_map = norm.mean() + loss_
    # print(norm.mean(), loss_)
    # loss[(1 - gtmaps).nonzero().squeeze()] = norm
    # loss[gtmaps.nonzero().squeeze()] = anom

    return score_map


def __norm_anom_loss(loss, gtmaps):
    norm = loss[gtmaps == 0]
    anom = loss[gtmaps == 1]
    if len(anom) == 0:
        anom = torch.zeros_like(norm)
    perm1 = torch.randperm(norm.size()[0], device=TrainConfigures.device)
    norm = norm[perm1]
    perm2 = torch.randperm(anom.size()[0], device=TrainConfigures.device)
    anom = anom[perm2]
    size = min(norm.size()[0], anom.size()[0])
    loss = 5 - (anom[: size] - norm[: size])
    # loss = 7.5 - (anom.mean()-norm.mean())
    loss[loss < 0] = 0
    # loss = abs(loss)
    return loss.mean()


def __norm_anom_margin(loss, gtmaps):
    norm = loss[gtmaps == 0]
    anom = loss[gtmaps == 1]
    size_n = norm.size()[0]
    size_a = anom.size()[0]
    r = int(size_n / size_a)
    anom = anom.repeat(r)
    norm = norm[:size_a * r]
    loss = 3.5 - (anom - norm)
    loss[loss < 0] = 0
    return loss.mean()


def __norm_anom_loss1(loss, gtmaps):
    norm = loss[gtmaps == 0]
    anom = loss[gtmaps == 1]
    perm1 = torch.randperm(norm.size()[0])
    norm = norm[perm1]
    perm2 = torch.randperm(anom.size()[0])
    anom = anom[perm2]
    size = min(norm.size()[0], anom.size()[0])
    loss = 3.5 - (anom[: size] - norm[: size])
    loss[loss < 0] = 0
    return loss.mean()


def __pixel_sim_loss(outs, labels):
    norm = outs[labels == 0]  # loss 张量对应于label张量元素为0的相应位置，形成的张量赋值给normal
    N, C, H, W = norm.size()
    B = int(N / 2)
    if N % 2 == 0:
        feat_p = norm[:B, :, :, :]
        feat_f = norm[B:, :, :, :]
    else:
        feat_p = norm[:B, :, :, :]
        feat_f = norm[B: -1, :, :, :]

    feat_p = F.normalize(feat_p, p=2, dim=1)
    feat_f = F.normalize(feat_f, p=2, dim=1)
    sim_dis = torch.tensor(0.).to(TrainConfigures.device)

    for i in range(1, B):
        for j in range(1, B):
            s_sim_map = pair_wise_sim_map(feat_p[i], feat_p[j])
            t_sim_map = pair_wise_sim_map(feat_f[i], feat_f[j])

            p_s = F.log_softmax(s_sim_map / 1.0 + 1e-31, dim=1)
            p_t = F.softmax(t_sim_map / 1.0, dim=1)

            sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
            sim_dis += sim_dis_
    sim_dis = sim_dis / B
    return sim_dis


def pair_wise_sim_map(fea_0, fea_1):
    C, H, W = fea_0.size()

    fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
    fea_1 = fea_1.reshape(C, -1).transpose(0, 1)

    sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))

    return sim_map_0_1


def pair_distance(fea_0):
    C, H, W = fea_0.size()
    P = H * W
    fea_0 = fea_0.reshape(C, P).unsqueeze(-1).expand(C, P, P) - fea_0.reshape(C, P).permute(1, 0).unsqueeze(0).expand(P, P, C).permute(2, 0, 1)
    fea_0 = -torch.norm(fea_0, p=2, dim=0)
    return fea_0


def pair_distance1(fea_0, s_0):
    P, C = fea_0.size()
    fea_0 = fea_0.permute(1, 0).unsqueeze(-1).expand(C, P, P) - s_0.unsqueeze(0).expand(P, P, C).permute(2, 0, 1)
    fea_0 = -torch.norm(fea_0, p=2, dim=0)
    return fea_0


def sim_distance(fea_0, scale1, scale2):
    C, H, W = fea_0.size()
    P = int(H / scale1) * int(W / scale2)
    # fea_0 = fea_0.reshape(C, int(H/2), 2, int(W/2), 2).permute(0, 1, 3, 2, 4).reshape(C, P, 4).permute(1, 2, 0).reshape(P, -1)
    fea_0 = fea_0.reshape(C, int(H / scale1), scale1, int(W / scale2), scale2).permute(0, 1, 3, 2, 4).reshape(C, P, scale1 * scale2).permute(1, 2, 0)
    _, _, C1 = fea_0.size()
    # fea_0 = fea_0.permute(1, 0).unsqueeze(-1).expand(C1, P, P) - fea_0.unsqueeze(0).expand(P, P, C1).permute(2, 0, 1)
    fea_0 = fea_0.permute(1, 2, 0).unsqueeze(-1).expand(scale1 * scale2, C1, P, P) - fea_0.unsqueeze(0).expand(P, P, scale1 * scale2, C1).permute(2, 3, 0, 1)
    fea_0 = -torch.norm(fea_0, p=2, dim=(0, 1))
    return fea_0


def sim_distance1(fea_0, scale1, scale2):
    C, H, W = fea_0.size()
    fea_0 = fea_0.reshape(C, int(H / scale1), scale1, int(W / scale2), scale2).permute(0, 1, 3, 2, 4).reshape(C, int(H / scale1), int(W / scale2), scale1 * scale2).permute(0, 3, 1,
                                                                                                                                                                            2).reshape(
        C * scale1 * scale2, int(H / scale1), int(W / scale2))
    fea_0 = F.normalize(fea_0, p=2, dim=0)
    fea_map = pair_wise_sim_map(fea_0, fea_0)
    return fea_map


def score1(outs):
    outs = torch.norm(outs, p=2, dim=1)
    loss = outs ** 2  # 张量间基本计算
    # loss = 1-(-((loss +1).sqrt() - 1)).exp()
    loss = (loss + 1).sqrt() - 1
    # loss = loss.sqrt()
    # loss = F.interpolate(loss, size=TrainConfigures.crop_size, mode='bilinear', align_corners=True)
    # loss = 1-(-loss).exp()
    loss = loss.squeeze()
    return loss


def score2(outs):
    # outs = F.interpolate(outs, size=TrainConfigures.crop_size, mode='bilinear', align_corners=True)
    outs = torch.norm(outs, p=2, dim=1)
    # outs = torch.sigmoid(outs)
    # outs = abs(outs)
    outs = outs.squeeze()
    return outs


def score3(outs, ref_mean, ref_var):
    outs = (outs - ref_mean) / ref_var
    outs = outs.squeeze()
    return abs(outs)


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


class ScoreObserver:
    def __init__(self, name, cls, min_train_epoch, max_train_epoch, threshold=1):
        self.name = name
        self.cls = cls
        self.threshold = threshold
        self.min_train_epoch = min_train_epoch
        self.max_train_epoch = max_train_epoch
        self.update_count = threshold

        self.last_epoch = 0
        self.last_score = 0.0

        self.max_epoch = 0
        self.max_score = 0.0

    def update(self, score, epoch, module):
        """
        update result score
        return True if max score haven't been changed after threshold times
        """

        logger.info(f'update: {self.name}/{self.cls}, epoch: {epoch}/{self.max_epoch}, score: {round(score, 2)}/{round(self.max_score, 2)}')

        self.last_epoch = epoch
        self.last_score = score
        if score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch

            if epoch > 0:
                torch.save(module.state_dict(), os.path.join(utils.get_dir(project_dir, 'output', 'modules'), f'{self.cls}_{"_".join([str(i) for i in TrainConfigures.crop_size])}.pth'))
                logger.info(f'update best result of {self.cls}, module saved')

            self.update_count = self.threshold
        elif epoch > self.min_train_epoch:
            self.update_count -= 1

        if self.update_count < 0 or epoch >= self.max_train_epoch or self.max_score == 100.:
            self.update_count = 1
            return True
        return False


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def simi_loss(outputs):
    # outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).squeeze()
    x1 = outputs.clone()
    x2 = outputs[torch.randperm(len(outputs))]
    # loss = 1/2*(l2_d(x1, x2)+l2_d(x2, x1))
    loss = D(x1, x2)

    return loss


def l2_d(p, z):
    z = z.detach()
    loss = p - z
    loss = torch.norm(loss, p=2, dim=1).mean()
    return loss


def D(p, z):
    z = z.detach()
    loss = 1 - ((F.normalize(p, dim=1) * F.normalize(z, dim=1)).sum(1))
    return loss.mean()


def mahal_score(train_output_in, test_output):
    N, C, H, W = test_output.size()
    test_output = test_output.reshape(N, C, H * W)
    dist_list = list()
    for i in range(H * W):
        mean = train_output_in[0][:, i]
        conv_inv = np.linalg.inv(train_output_in[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in test_output]
        dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(N, H, W)
    dist_list_tensor = torch.tensor(dist_list)
    return dist_list_tensor


def L2_score(u, test_output):
    N, C, H, W = test_output.size()
    u = u.reshape(1, C, H, W)
    score = test_output - u
    score = torch.norm(score, p=2, dim=1)

    return score


def count_non_zeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
    return total, nonzero


if __name__ == '__main__':
    main()
