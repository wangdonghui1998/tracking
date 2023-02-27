import cv2

from . import BaseActor
import torch
import matplotlib.pyplot as plt
import visdom
import numpy as np
from pytracking.utils.get_samples import get_samples
import math
from torchvision import transforms
from matplotlib import patches
from PIL import Image
from matplotlib.ticker import MultipleLocator
vis = visdom.Visdom(port=8098)

class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class LSDiMPActor(BaseActor):
    """ Actor for training LSDiMP model """

    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_fn = dimp_jitter_fn
        # TODO set it somewhere
        self.device = torch.device("cuda:1")

    def max2d(self,a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Computes maximum and argmax in the last two dimensions."""

        max_val_row, argmax_row = torch.max(a, dim=-2)
        max_val, argmax_col = torch.max(max_val_row, dim=-1)
        argmax_row = argmax_row.view(argmax_col.numel(), -1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
        argmax_row = argmax_row.reshape(argmax_col.shape)
        argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
        return max_val, argmax

    def gauss_2d(self,sz,sigma, center, end_pad=(0, 0), density=False):
        # sz=18, sigma=0.9, center, end_pad=（1，1）
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        return self.gauss_1d(sz[0].item(), sigma[0], center[:,0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
               self.gauss_1d(sz[1].item(), sigma[1], center[:,1], end_pad[1], density).reshape(center.shape[0], -1, 1)

    def gauss_1d(self,sz, sigma, center, end_pad=0, density=False):
        # sz = 18, sigma = 0.9, center, end_pad =（0，0）,
        # k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1).to(self.device)
        k = torch.arange(0, sz).reshape(1, -1).to(self.device)
        gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
        if density:
            gauss /= math.sqrt(2 * math.pi) * sigma
        return gauss

    #直接进行插值
    def get_loc1(self, crop_box, score, sample_scales,img_sample_sz):
        max_score, max_disp = self.max2d(score)
        max_disp = max_disp.squeeze()
        crop_box = crop_box.squeeze(0)     # [4]
        crop_x, crop_y, crop_w, crop_h = crop_box.tolist()
        max_disp = max_disp * (sample_scales * img_sample_sz / score.shape[-1])
        max_disp[0] = max_disp[0] + crop_x
        max_disp[1] = max_disp[1] + crop_y
        return max_score,max_disp

    def get_loc_delta(self, crop_box, max_disp, delta, sample_scales, img_sample_sz, sz):
        #crop_box[10,4],max_disp[10,2],delta[10,2],sample_scales[10],img_sample_sz=388,sz=18
        max_disp = delta + torch.tensor(max_disp).to(self.device)     #[10,2]
        crop_x,crop_y,crop_w,crop_h = crop_box[:,0],crop_box[:,1],crop_box[:,2],crop_box[:,3]   #都是[10]
        max_disp[:,0] = max_disp[:,0] - crop_x
        max_disp[:,1] = max_disp[:,1] - crop_y
        max_disp[:,0] = max_disp[:,0] / (sample_scales * img_sample_sz / sz)  #映射回score map 空间
        max_disp[:,1] = max_disp[:,1] / (sample_scales * img_sample_sz / sz)
        score_map_max = torch.full_like(max_disp, sz-1)
        max_disp = torch.min(max_disp,score_map_max)  #[10,2]
        # gauss
        if isinstance(sz, (float, int)):
            sz = torch.tensor([sz, sz])
        sigma_factor = 0.055469755418553784
        sigma = sigma_factor * sz.prod().sqrt().item()
        lstm_score_map = self.gauss_2d(sz=sz, sigma = sigma, center=max_disp)  # 基于预测出来的位置的guass

        return lstm_score_map

    def draw_deltaxy(self,num_squeeze,lstm_target,lstm_pred):
        x = MultipleLocator(5)  # x轴每5个一个刻度
        ax = plt.gca()
        ax.xaxis.set_major_locator(x)
        order = np.arange(0, lstm_target.shape[0], 1)
        gt_eudistance = np.zeros(lstm_target.shape[0])
        pred_eudistance = np.zeros(lstm_target.shape[0])
        lstm_target = lstm_target.squeeze().cpu().detach().numpy()  # (n,2)
        lstm_pred = lstm_pred.cpu().detach().numpy()  # (n,2)

        for i in range(0, lstm_target.shape[0]):
            gt_eudistance[i] = np.sqrt(lstm_target[i,0] ** 2 + lstm_target[i,0] ** 2)
            pred_eudistance[i] = np.sqrt(lstm_pred[i,0] ** 2 + lstm_pred[i,0] ** 2)
        #画target和pred的欧氏距离
        plt.title(num_squeeze)
        plt.plot(order,gt_eudistance,color="b")
        plt.plot(order,pred_eudistance,color="r")
        vis.matplot(plt)
        plt.close()


    def __call__(self, data,optimizer,loader):
        dimp_train_imgs = data['train_images'].to(self.device)  # [2,10,3,288,288]  10是batch_size
        dimp_train_bb = data['train_anno'].to(self.device)  # [2,10,4]   train的bbox坐标
        lstm_crops_boxes = data['test_crop_boxes']  #[30,10,4]
        lstm_real_imgs = data['test_real_images']
        lstm_real_anno = data['test_real_anno']      #[30,10,4]
        lstm_sample_scales = data['test_im_resize_factor']  #[30,10]

        sequence_length = data['test_images'].shape[0]  # 用于lstm训练的帧数   30
        num_sequences = data['test_images'].shape[1]  # 一次性输进来的序列的个数 batch_size  10
        img_sample_sz = dimp_train_imgs.shape[-1]  # 目前图片的大小 288

        #batch size first
        lstm_crops_boxes = lstm_crops_boxes.permute(1, 0, 2)
        lstm_sample_scales = lstm_sample_scales.permute(1, 0)

        # 存储score map
        max_score_all = np.zeros((0, 1))  # 30 , 1
        max_disp_all = np.zeros((0, 2))  # 30 , 2
        in_seq_len = 5
        pred_len = 1
        delta_xy_disp = torch.zeros((num_sequences, sequence_length-1, 2))  #10, 29, 2

        # 各种loss的声明
        lstm_clf_loss_all = torch.zeros(num_sequences, sequence_length).to(self.device)    #[10,30]
        dimp_clf_loss_all = torch.zeros(num_sequences, sequence_length).to(self.device)    #[10,30]
        bbreg_loss_all = torch.zeros(num_sequences, sequence_length).to(self.device)
        lstm_loss_all = torch.zeros(num_sequences).to(self.device)    #[10]

        # acc声明
        lstm_clf_acc = 0
        dimp_clf_acc = 0

        #初始化外观模型
        dimp_filters = self.net.train_classifier(dimp_train_imgs, dimp_train_bb)  # [10,512,4,4]

        #目标中心点位置
        lstm_real_anno = lstm_real_anno.permute(1, 0, 2)  # batch_size first
        center_x = lstm_real_anno[:, :, 0] + lstm_real_anno[:, :, 2] / 2
        center_y = lstm_real_anno[:, :, 1] + lstm_real_anno[:, :, 3] / 2
        center_x = center_x.unsqueeze(-1)
        center_y = center_y.unsqueeze(-1)
        lstm_center = torch.cat([center_x, center_y], 2).to(self.device)   #[10,30,2]

        #存储目标偏移量
        for i in range(0, sequence_length):
            if i >= 1:  # 从第二帧开始，存储偏移量
                delta = lstm_center[:, i, :] - lstm_center[:, i - 1, :]
                delta_xy_disp[:, i - 1, :] = delta  # 存储偏移量，偏移量序号比当前帧序号少1

        #目标预测出来的位置
        pred_delta_xy = torch.zeros((num_sequences, sequence_length-in_seq_len-1, 2)).to(self.device)  #10, 24, 2
        # # 将坐标数据送去lstm中训练
        for i in range(0,num_sequences):
            lstm_data = delta_xy_disp[i]
            lstm_train, lstm_target = get_samples(lstm_data,in_seq_len=in_seq_len,pred_len=pred_len,device=self.device)
            lstm_pred = self.net.lstm_net(lstm_train)   #[24,2]
            self.draw_deltaxy(num_sequences,lstm_target,lstm_pred)
            lstm_loss = self.objective['lstm'](lstm_pred, lstm_target)  #1个序列的loss
            if loader.training:
                optimizer.zero_grad()
                lstm_loss.backward()
                optimizer.step()
            lstm_loss_all[i] = lstm_loss
            pred_delta_xy[i] = lstm_pred


        #开始跟踪训练lstm的序列,前五帧都预测不到
        # for i in range(0, sequence_length):
        #     lstm_image_cur = data['test_images'][i, ...].to(self.device)   #[10,3,288,288]    10是batch_size
        #     lstm_label_cur = data['test_label'][i:i + 1, ...].to(self.device)   #[1,10,19,19]
        #     lstm_label_cur = lstm_label_cur[:, :, :-1, :-1].contiguous()  # [1,10,18,18] contiguous() 深拷贝，两个变量没有关系
        #     lstm_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)  #[1,1,4]
        #
        #     # Extract features  提取特征
        #     backbone_feat_cur_all = self.net.extract_backbone_features(lstm_image_cur)
        #     backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]    #[10,1024,18,18]
        #     backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
        #                                                backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])  #[1,10,1024,18,18]
        #
        #     # Run target model 外观模型得分
        #     dimp_target_scores = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)   #[1,10,19,19]
        #     dimp_target_scores = dimp_target_scores[:, :, :-1, :-1].contiguous()  #[1,10,18,18]
        #
        #     # Jitter target model output for augmentation
        #     jitter_info = None
        #     if self.dimp_jitter_fn is not None:
        #         dimp_target_scores = self.dimp_jitter_fn(dimp_target_scores, lstm_label_cur.clone())
        #
        #
        #     # real_anno = lstm_real_anno[i].squeeze()
        #     # fig, ax = plt.subplots(1)  # fig,（空格）必须要，没有ax无法调用add_patch
        #     # plt.title("real_imgs" + str(i))
        #     # ax.imshow(lstm_real_imgs[i].squeeze().cpu().detach().numpy(),cmap='viridis')
        #     # # Rectangle 坐标的参数格式为左上角（x, y），width, height。
        #     # rec = patches.Rectangle((real_anno[0], real_anno[1]), real_anno[2], real_anno[3], linewidth=2, edgecolor='r', facecolor='None')
        #     # ax.add_patch(rec)
        #     # # ax.plot(max_disp[0], max_disp[1],'o', color="r")
        #     # ax.plot(lstm_center[i][0].cpu(),lstm_center[i][1].cpu(),'o', color="r")
        #     # vis.matplot(plt)
        #     # print()
        #     # plt.close()
        #     #
        #     # plt.title("dimp_target_scores" + str(i))
        #     # img = dimp_target_scores[0][0]
        #     # plt.imshow(img.cpu().detach().numpy(), cmap='viridis')
        #     # vis.matplot(plt)
        #     # print()
        #     #
        #     # plt.title("dimp_label_cur" + str(i))
        #     # img = lstm_label_cur[0][0]
        #     # plt.imshow(img.cpu().detach().numpy(), cmap='viridis')
        #     # vis.matplot(plt)
        #     # print()
        #
        #     #使用偏移量做预测
        #     # if i >= 1:   #从第二帧开始，存储偏移量
        #     #     delta = lstm_center[:, i, :] - lstm_center[:, i - 1, :]
        #     #     delta_xy_disp[:, i-1, :] = delta  # 存储偏移量，偏移量序号比当前帧序号少1
        #     #     lstm_data = delta_xy_disp[:,0:i,:]  #[10,1，2] -> [10,2,2]
        #
        #     #使用欧氏距离做预测
        #
        #     # lstm预测不到的前几帧默认为tomp score map的结果
        #     if i < 6:    #前 0，1，2，3，4，5 都预测不到，用作训练，因为采用delta xy预测，即采用前5个帧间插值
        #         target_scores = dimp_target_scores   #[1,10,18,18]
        #     else:   #从id = 7 开始才可以运用lstm进行预测
        #
        #         # 生成基于lstm预测结果的高斯结果图
        #         lstm_score_map = self.get_loc_delta(lstm_crops_boxes[:,i,:], lstm_center[:,i-1,:], pred_delta_xy[:,i-in_seq_len-1,:],
        #                                             lstm_sample_scales[:,i], img_sample_sz, dimp_target_scores.shape[-1])
        #         lstm_score_map = lstm_score_map.unsqueeze(0)
        #
        #         # plt.title("lstm_target_scores" + str(i))
        #         # img = lstm_score_map[0][0]
        #         # plt.imshow(img.cpu().detach().numpy(), cmap='viridis')
        #         # vis.matplot(plt)
        #         # print()
        #
        #
        #         # 将两个分类得分进行加权融合，得出最后的score map
        #         score_map1 = 0.5 * dimp_target_scores + 0.5 * lstm_score_map
        #         target_scores = score_map1
        #         # plt.title("target_scores" + str(i))
        #         # img = target_scores[0][0]
        #         # plt.imshow(img.cpu().detach().numpy(), cmap='viridis')
        #         # vis.matplot(plt)
        #         # print()
        #
        #     # #计算loss之和，然后在循环外取平均
        #     # lstm的loss
        #
        #     # dimp的分类的loss
        #     dimp_clf_loss_test = self.objective['dimp_clf'](dimp_target_scores, lstm_label_cur,lstm_anno_cur)
        #     dimp_clf_loss_all[:, i] = dimp_clf_loss_test
        #
        #     # 新的分类分支 loss
        #     # # Classification losses for the different optimization iterations
        #     lstm_clf_loss_test = self.objective['lstm_clf'](target_scores,lstm_label_cur,lstm_anno_cur)
        #     lstm_clf_loss_all[:, i] = lstm_clf_loss_test
        #
        #     # 准确度 acc
        #     dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_target_scores,
        #                                                                     lstm_label_cur,
        #                                                                     valid_samples=torch.tensor([[1]]).to(
        #                                                                         self.device))
        #     dimp_clf_acc += dimp_clf_acc_new
        #
        #     lstm_clf_acc_new, lstm_pred_correct = self.objective['clf_acc'](target_scores,
        #                                                                     lstm_label_cur,
        #                                                                     valid_samples=torch.tensor([[1]]).to(
        #                                                                         self.device))
        #     lstm_clf_acc += lstm_clf_acc_new

        # #整体loss
        lstm_loss_mean = lstm_loss_all.mean()
        # lstm_clf_loss_mean = lstm_clf_loss_all.mean()
        # dimp_clf_loss_mean = dimp_clf_loss_all.mean()
        # loss = self.loss_weight['dimp_clf'] * dimp_clf_loss_mean  + \
        #        self.loss_weight['lstm_clf'] * lstm_clf_loss_mean + \
        #        self.loss_weight['lstm'] * lstm_loss_mean
        loss = self.loss_weight['lstm'] * lstm_loss_mean   # 改后的loss
        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        #整体acc
        dimp_clf_acc /= sequence_length
        lstm_clf_acc /= sequence_length

        stats = {'Loss/total': loss.item(),
                 'Loss/lstm': lstm_loss_mean.item(),
                 # 'Loss/lstm_clf': lstm_clf_loss_mean.item(),
                 # 'Loss/dimp_clf': dimp_clf_loss_mean.item(),
                 # 'Loss/raw/lstm_clf_acc': lstm_clf_acc.item(),
                 # 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item()
                 }

        return loss, stats

class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                            loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class KYSActor(BaseActor):
    """ Actor for training KYS model """
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_fn = dimp_jitter_fn

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Initialize loss variables
        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        # Initialize target model using the training frames   使用训练帧初始化目标模型
        train_images = data['train_images'].to(self.device)   #image [3,1,3,288,288]
        train_anno = data['train_anno'].to(self.device)    #bbox信息  [3,1,4]
        dimp_filters = self.net.train_classifier(train_images, train_anno)  #[1,512,4,4]

        # Track in the first test frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col (added due to even kernel size in the target model)
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        # Set previous frame information
        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        # Loop over the sequence
        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()   #contiguous() 深拷贝，两个变量没有关系
            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            # Extract features
            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all).view(1, num_sequences, -1,
                                                                                             backbone_feat_cur.shape[-2],
                                                                                             backbone_feat_cur.shape[-1])
            else:
                motion_feat_cur = backbone_feat_cur

            # Run target model
            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            # Jitter target model output for augmentation
            jitter_info = None
            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

            # Input target model output along with previous frame information to the predictor
            predictor_input_data = {'input1': motion_feat_prev, 'input2': motion_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame

            # Calculate losses
            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_wight.keys():
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur, test_anno_cur,  valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * ((test_pred_correct).long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_samples=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = clf_loss_test_all.mean()
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf':    clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats


class DiMPSimpleActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'],
                                            train_label=data['train_label'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        loss_test_init_clf = 0
        loss_test_iter_clf = 0

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Loss for the initial filter iteration
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class TargetCandiateMatchingActor(BaseActor):
    """Actor for training the KeepTrack network."""
    def __init__(self, net, objective):
        super().__init__(net, objective)

    def __call__(self, data):
        """
        args:
            data - The input data.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        preds = self.net(**data)

        # Classification losses for the different optimization iterations
        losses = self.objective['target_candidate_matching'](**data, **preds)


        # Total loss
        loss = losses['total'].mean()

        # Log stats
        stats = {
            'Loss/total': loss.item(),
            'Loss/nll_pos': losses['nll_pos'].mean().item(),
            'Loss/nll_neg': losses['nll_neg'].mean().item(),
            'Loss/num_matchable': losses['num_matchable'].mean().item(),
            'Loss/num_unmatchable': losses['num_unmatchable'].mean().item(),
            'Loss/sinkhorn_norm': losses['sinkhorn_norm'].mean().item(),
            'Loss/bin_score': losses['bin_score'].item(),
        }

        if hasattr(self.objective['target_candidate_matching'], 'metrics'):
            metrics = self.objective['target_candidate_matching'].metrics(**data, **preds)

            for key, val in metrics.items():
                stats[key] = torch.mean(val[~torch.isnan(val)]).item()

        return loss, stats


class ToMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g) # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):  #类重载()，以对象名()形式使用
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        # 此处调用tompnet.forward()函数
        target_scores, bbox_preds = self.net(train_imgs=data['train_images'],    #[2,1,3,288,288]
                                             test_imgs=data['test_images'],     #[1,1,3,288,288]
                                             train_bb=data['train_anno'],      #[2,1,4]
                                             train_label=data['train_label'],   #[2,1,18,18] 目标位置编码，类似置信度图
                                             train_ltrb_target=data['train_ltrb_target'])   #[2,1,4,18,18] ltrb编码：特征图上每个位置距离bbox的归一化距离

        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])   #调用 GIoULoss()的forward(预测目标，真实取样区域)函数

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test


        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou']*loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf']*clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious>0].std().item()

        return loss, stats

class LSToMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g) # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def max2d(self,a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Computes maximum and argmax in the last two dimensions."""

        max_val_row, argmax_row = torch.max(a, dim=-2)
        max_val, argmax_col = torch.max(max_val_row, dim=-1)
        argmax_row = argmax_row.view(argmax_col.numel(), -1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
        argmax_row = argmax_row.reshape(argmax_col.shape)
        argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
        return max_val, argmax

    def gauss_2d(self,sz, center, end_pad=(0, 0), density=False):
        # sz=18, sigma=0.9, center, end_pad=（0，0）,
        scores = np.zeros((sz, sz))
        if isinstance(sz, (float, int)):
            sz = torch.tensor([sz, sz])
        scores[int(center[0].item())][int(center[1].item())] = 1
        sigma = scores.std()
        sigma = sigma * sz.prod().sqrt().item()
        center = center.unsqueeze(0)
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        return self.gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
               self.gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)

    def gauss_1d(self,sz, sigma, center, end_pad=0, density=False):
        # sz = 18, sigma = 0.9, center, end_pad =（0，0）,
        # k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
        k = torch.arange(0, sz).reshape(1, -1).to(self.device)
        gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
        if density:
            gauss /= math.sqrt(2 * math.pi) * sigma
        return gauss

    def direct_bbox_regression(self, bbox_preds, sample_coords, score_loc, scores_raw):
        shifts_x = torch.arange(
            0, self.img_sample_sz[0], step=16,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            0, self.img_sample_sz[1], step=16,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + 16 // 2
        xs, ys = locations[:, 0], locations[:, 1]
        s1, s2 = scores_raw.shape[2:]
        xs = xs.reshape(s1, s2)
        ys = ys.reshape(s1, s2)

        ltrb = bbox_preds.permute(0,1,3,4,2)[0,0].cpu() * self.train_img_sample_sz[[0, 1, 0, 1]]
        xs1 = xs - ltrb[:, :, 0]
        xs2 = xs + ltrb[:, :, 2]
        ys1 = ys - ltrb[:, :, 1]
        ys2 = ys + ltrb[:, :, 3]
        sl = score_loc.int()

        x1 = xs1[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y1 = ys1[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        x2 = xs2[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y2 = ys2[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        w = x2 - x1
        h = y2 - y1

        return torch.Tensor([x1, y1, w, h])

    #直接进行插值
    def get_loc1(self,shift_box,score,output_size,im_resize_factor):

        score = score.squeeze(0).permute(2,1,0).cpu().detach().numpy()
        shift_box = shift_box.squeeze(0)
        resize_factor = math.ceil(output_size * im_resize_factor.item())
        # #双三次插值法获得图像大小
        oral_score = cv2.resize(score,(resize_factor, resize_factor),Image.BICUBIC)
        oral_score = torch.tensor(oral_score).unsqueeze(0)

        shift_x,shift_y,shift_w,shift_h = shift_box.tolist()
        # 找出score中的最大值点和对应的坐标
        max_score,max_disp = self.max2d(oral_score)
        max_disp[0][0] = max_disp[0][0] + shift_x
        max_disp[0][1] = max_disp[0][1] + shift_y
        return max_score,max_disp

    def get_loc2(self,shift_box,pred,output_size,im_resize_factor):
        shift_x, shift_y, shift_w, shift_h = shift_box.tolist()
        pred[0] = pred[0] - shift_x
        pred[1] = pred[1] - shift_y

        return pred


    def __call__(self, data):  #类重载()，以对象名()形式使用
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        # 此处调用tompnet.forward()函数
        tomp_train_imgs = data['train_images']     #[2,1,3,288,288]
        lstm_train_imgs = data['test_images']       #[30,1,3,288,288]
        train_bb = data['train_anno']         #[2,1,4]   train的bbox坐标
        test_bb = data['test_anno']           #[30,1,4]  test的bbox坐标
        test_shift_boxes = data['test_shift_boxes']
        test_im_resize_factor = data['test_im_resize_factor']
        train_label = data['train_label']     #[2,1,18，18]  目标位置编码，类似置信度图
        train_ltrb_target = data['train_ltrb_target']   #[2,1,4，18，18]  ltrb编码：特征图上每个位置距离bbox的归一化距离

        sequence_length = data['test_images'].shape[0]  #用于lstm训练的帧数
        num_sequences = data['test_images'].shape[1]    #序列的个数  1
        crop_img_size = tomp_train_imgs.shape[-1]     #目前图片的大小 288

        #存储score map
        max_score_all = np.zeros((0, 1))   #30 , 1
        max_disp_all = np.zeros((0, 2))    #30 , 2

        #各种loss的声明
        lstm_clf_loss_all =  torch.zeros(num_sequences, sequence_length).to(self.device)
        tomp_clf_loss_all = torch.zeros(num_sequences, sequence_length).to(self.device)
        bbreg_loss_all =  torch.zeros(num_sequences, sequence_length).to(self.device)
        lstm_loss_all =  torch.zeros(num_sequences, sequence_length).to(self.device)

        #acc声明
        lstm_clf_acc = 0
        tomp_clf_acc = 0

        #tomp每两帧获取分类和回归权重和后一帧测试增强测试帧
        for i in range(0,sequence_length):
            tomp_cls_filter, tomp_breg_filter, tomp_test_feat_enc = self.net.tomp_train_classifier(tomp_train_imgs,
                                                                                lstm_train_imgs[i],
                                                                                train_bb,
                                                                                train_label,
                                                                                train_ltrb_target)
            # cls_filter:[1,256,1,1], breg_filter:[1,256,1,1], test_feat_enc:[1,1,256,18,18]

            #计算基于tomp的lstm帧的分类得分 和 bbox_pred 预测
            tomp_target_scores = self.net.tomp_head.classifier(tomp_test_feat_enc, tomp_cls_filter)   #[1,1,18,18]
            tomp_bbox_preds = self.net.tomp_head.bb_regressor(tomp_test_feat_enc, tomp_breg_filter)  #[1,1,4,18,18]

            # plt.title("tomp_target_scores"+str(i))
            # img = tomp_target_scores[0][0]
            # plt.imshow(img.cpu().detach().numpy(), cmap='viridis')
            # vis.matplot(plt)
            # print()

            #找到max_score中的高亮点在原图中的位置
            max_score,max_disp = self.get_loc1(test_shift_boxes[i],tomp_target_scores, crop_img_size,test_im_resize_factor[i])
            max_score = max_score.squeeze(0).cpu().detach().numpy()
            max_disp = max_disp.squeeze().cpu().detach().numpy()

            max_score_all = np.append(max_score_all,max_score)
            max_disp_all = np.append(max_disp_all,[[max_disp[0],max_disp[1]]],axis=0)

            # self.draw(tomp_target_scores,data['test_real_images'][i],data['test_real_anno'][i],max_disp)

            # lstm预测不到的前几帧默认为tomp score map的结果
            if i < 5:
                target_scores = tomp_target_scores
                lstm_loss = 0
            else:
                # 加入lstm模块
                n_steps_in = 5
                n_steps_out = 1
                lstm_train, lstm_target = get_samples(max_disp_all, n_steps_in, n_steps_out,self.device)
                lstm_pred = self.net.lstm_net(lstm_train) #获取预测出来的pred
                lstm_target = lstm_target[-1] #只取最后一个target值
                lstm_loss = self.objective['lstm_clf'](lstm_pred.unsqueeze(0),lstm_target)

                # 生成基于lstm预测结果的高斯结果图
                lstm_pred = lstm_pred.round()
                lstm_pred = self.get_loc2(test_shift_boxes[i],tomp_target_scores, crop_img_size,test_im_resize_factor[i])
                lstm_score_map = self.gauss_2d(sz = tomp_target_scores.shape[2],center=lstm_pred)
                lstm_score_map = lstm_score_map.view(-1,1,18,18)

                # 将两个分类得分进行加权融合，得出最后的score map
                score_map1 = 0.5 * tomp_target_scores + 0.5 * lstm_score_map
                target_scores = score_map1


            # #计算loss之和，然后在循环外取平均
            #lstm的loss
            lstm_loss_all[:, i] = lstm_loss

            #tomp的分类的loss
            tomp_clf_loss_test = self.objective['test_clf'](tomp_target_scores, data['test_label'][i].unsqueeze(0), data['test_anno'][i])
            tomp_clf_loss_all[:, i] = tomp_clf_loss_test

            #新的分类分支 loss
            # # Classification losses for the different optimization iterations
            lstm_clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'][i].unsqueeze(0), data['test_anno'][i])
            lstm_clf_loss_all[:, i] = lstm_clf_loss_test
            #
            # #tomp bbreg 分支 loss
            tomp_loss_giou, ious = self.objective['tomp_giou'](tomp_bbox_preds, data['test_ltrb_target'][i].unsqueeze(0),data['test_sample_region'][i].unsqueeze(0))  # 调用 GIoULoss()的forward(预测目标，真实取样区域)函数
            bbreg_loss_all[:, i] = tomp_loss_giou

            #准确度 acc
            tomp_clf_acc_new, tomp_pred_correct = self.objective['clf_acc'](tomp_target_scores, data['test_label'][i].unsqueeze(0),
                                                                            valid_samples=torch.tensor([[1]]).to(self.device))
            tomp_clf_acc += tomp_clf_acc_new

            lstm_clf_acc_new, lstm_pred_correct = self.objective['clf_acc'](target_scores, data['test_label'][i].unsqueeze(0),
                                                                            valid_samples=torch.tensor([[1]]).to(self.device))
            lstm_clf_acc += lstm_clf_acc_new

        # #整体loss
        lstm_loss_mean = lstm_loss_all.mean()
        lstm_clf_loss_mean = lstm_clf_loss_all.mean()
        tomp_clf_loss_mean = tomp_clf_loss_all.mean()
        bbreg_loss_mean = bbreg_loss_all.mean()

        loss = self.loss_weight['tomp_giou'] * bbreg_loss_mean + \
               self.loss_weight['test_clf'] * lstm_clf_loss_mean + \
               self.loss_weight['lstm'] * lstm_loss_mean
        # loss = self.loss_weight['tomp_giou'] * bbreg_loss_mean + self.loss_weight['test_clf'] * tomp_clf_loss_mean + self.loss_weight['lstm'] * lstm_clf_loss_mean + lstm_loss_mean

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        #整体acc
        tomp_clf_acc /= sequence_length
        lstm_clf_acc /= sequence_length

        # ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], tomp_bbox_preds)
        #
        # stats = {'Loss/total': loss.item(),
        #          'Loss/GIoU': loss_giou.item(),
        #          'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
        #          'Loss/clf_loss_test': clf_loss_test.item(),
        #          'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
        #          'mIoU': ious.mean().item(),
        #          'maxIoU': ious.max().item(),
        #          'minIoU': ious.min().item(),
        #          'mIoU_pred_center': ious_pred_center.mean().item()
        #          }
        #
        # if ious.max().item() > 0:
        #     stats['stdIoU'] = ious[ious > 0].std().item()

        stats = {'Loss/total': loss.item(),
                 'Loss/lstm': lstm_loss_mean.item(),
                 'Loss/lstm_clf': lstm_clf_loss_mean.item(),
                 'Loss/bbreg': bbreg_loss_mean.item(),
                 'Loss/tomp_clf': tomp_clf_loss_mean.item(),
                 'Loss/raw/lstm_clf_acc': lstm_clf_acc.item(),
                 'Loss/raw/tomp_clf_acc': tomp_clf_acc.item()
                 }


        return loss, stats