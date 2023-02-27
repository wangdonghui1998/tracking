from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
from ltr.models.layers import activation
from lstm_model.lstm_series_model import LSTM,BiLSTM,CNN_LSTM
import visdom
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict


class LSToMP(BaseTracker):

    multiobj_mode = 'parallel'      #训练中mode是

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network  加载network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # The LSTM network
        self.lstm_net = CNN_LSTM(in_channels=2, out_channels=32, kernel_size=3, hidden_size=64, num_layers=1,
                                 output_size=2).to(self.params.device)
        self.lstm_net.load_state_dict(torch.load(self.params.lstm_net_path))

        self.center = None

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)    #[1，3，1080,1920]

        # Get target position and size
        state = info['init_bbox']      # 第一帧的bbox坐标
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])   # bbox中心点坐标————目标为中心的高斯分布
        self.target_sz = torch.Tensor([state[3], state[2]])     #目标大小————目标大小ltrb编码

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]     #None
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)  # ""

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])    #图像大小 [1080,1920]
        sz = self.params.image_sample_size            #图像采样大小，288
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)    #[288,288]
        if self.params.get('use_image_aspect_ratio', False):   #是否使用图像纵横比 ,False
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz           #图像采样大小 [288,288]
        self.img_support_sz = self.img_sample_sz   #图像支持的大小？？？？  [288,288]
        tfs = self.params.get('train_feature_size', 18)     #训练特征的大小  18
        stride = self.params.get('feature_stride', 16)      #特征stride（步幅） 16
        self.train_img_sample_sz = torch.Tensor([tfs*stride, tfs*stride])   #训练特征大小[288,288]

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()    #图像搜索区域  321900
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()    #目标相对于搜索区域的比例  1.97

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale    #初始帧目标的大小 [56.3449,58.883] ————目标大小ltrb编码

        # Setup scale factors 设置比例因子
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)   #tensor([1.])
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds   #设置比例边界
        self.min_scale_factor = torch.max(10 / self.base_target_sz)     #0.1775
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)     #19.1676

        # Extract and transform sample  数据增强 + 图像采样 + resnet 初始特征提取
        init_backbone_feat = self.generate_init_samples(im)   #[1,1024,18,18]

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        self.logging_dict = defaultdict(list)

        self.target_scales = []
        self.target_not_found_counter = 0

        self.cls_weights_avg = None

        #存储目标坐标
        self.max_disp_all = np.zeros((0, 2))

        # out = {'time': time.time() - tic}
        out = {'target_bbox': state}

        return out

    def clip_bbox_to_image_area(self, bbox, image, minwidth=10, minheight=10):
        H, W = image.shape[:2]
        x1 = max(0, min(bbox[0], W - minwidth))
        y1 = max(0, min(bbox[1], H - minheight))
        x2 = max(x1 + minwidth, min(bbox[0] + bbox[2], W))
        y2 = max(y1 + minheight, min(bbox[1] + bbox[3], H))
        return torch.Tensor([x1, y1, x2 - x1, y2 - y1])

    def encode_bbox(self, bbox):
        stride = self.params.get('feature_stride')
        output_sz = self.params.get('image_sample_size')

        shifts_x = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shifts_y = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        xs, ys = locations[:, 0], locations[:, 1]

        xyxy = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2],
                            bbox[:, 1] + bbox[:, 3]], dim=1)

        l = xs[:, None] - xyxy[:, 0][None]
        t = ys[:, None] - xyxy[:, 1][None]
        r = xyxy[:, 2][None] - xs[:, None]
        b = xyxy[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

        reg_targets_per_im = reg_targets_per_im / output_sz

        sz = output_sz // stride
        nb = bbox.shape[0]
        reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, nb, 4).permute(2, 3, 0, 1)

        return reg_targets_per_im

    def track(self, image, info: dict = None, prev_output:dict=None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)  #[1,3,1080,1920]

        # ------- LOCALIZATION ------- #

        # Extract backbone features  resnet 提取主干特征，patches块，patches块坐标
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)    #dict 类型[1,1024,18,18]  im_patches:torch.Size([1, 3, 288, 288])
        # Extract classification features 获取分类特征（用resnet layer3的特征）
        test_x = self.get_backbone_head_feat(backbone_feat)     #[1,1024,18,18]

        # Location of sample 采样的位置
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores 获取基于tomp的分类得分和回归分支 --- 回归分支：上下左右几个框的大致位置
        scores_raw, bbox_preds = self.classify_target(test_x)  #tomp_scores_raw[1,1,18,18]   tomp_bbox_preds:[1,1,4,18,18]

        #加上lstm进行修正
        if self.frame_num > 6:
            # 获取 delta x ,delta y
            delta_xy = prev_output['delta_xy'] #list 列表
            delta_xy = torch.tensor([item for item in delta_xy]).to(self.params.device)
            n_steps_in = 5
            n_steps_out = 1
            lstm_train, lstm_target = self.get_samples(delta_xy, n_steps_in, n_steps_out, self.params.device)
            lstm_pred = self.lstm_net(lstm_train)  # 获取预测出来的delta_xy
            # if self.frame_num == 80 :
            #     self.draw_deltaxy(lstm_target,lstm_pred)
            lstm_pred = lstm_pred[-1].unsqueeze(0)


            #将预测值加入上一帧center值得到当前帧的center值
            self.center = lstm_pred.cpu() + torch.tensor(prev_output['center'][-1]).squeeze(0)
            #将center值映射回18 * 18区域
            self.center[:,0] = self.center[:,0] - sample_coords[0, 1]
            self.center[:,1] = self.center[:,1] - sample_coords[0, 0]
            self.center = self.center /(sample_scales * self.img_sample_sz/self.feature_sz )

        translation_vec, scale_ind, s, flag, score_loc = self.localize_target(scores_raw, sample_pos, sample_scales, self.center)

        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)   #tomp_bbox_pred，patch块坐标，得分最大值点，得分图
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)     #bbox_raw,原图

        if flag != 'not_found':
            self.pos = bbox[:2].flip(0) + bbox[2:].flip(0)/2  # [y + h/2, x + w/2]
            self.target_sz = bbox[2:].flip(0)
            self.target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())
            self.target_scales.append(self.target_scale)
        else:
            if self.params.get('search_area_rescaling_at_occlusion', False):
                self.search_area_rescaling()

        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False) and scores_raw.max() > self.params.get('conf_ths', 0.0):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])
            train_y = self.get_label_function(self.pos, sample_pos[scale_ind, :], sample_scales[scale_ind]).to(
                self.params.device)

            # Update the classifier model
            self.update_memory(TensorList([train_x]), train_y, target_box, learning_rate)

        score_map = s[scale_ind, ...]


        # Compute output bounding box
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[0,[1,0]], sample_coords[0,[3,2]] - sample_coords[0,[1,0]] - 1))

        if self.params.get('output_not_found_box', False):
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state,
               'object_presence_score': score_map.max().cpu().item()}

        if self.visdom is not None:
            self.visualize_raw_results(score_map)

        return out

    def visualize_raw_results(self, score_map):
        self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
        self.logging_dict['max_score'].append(score_map.max())
        self.visdom.register(torch.tensor(self.logging_dict['max_score']), 'lineplot', 3, 'Max Score')
        self.debug_info['max_score'] = score_map.max().item()
        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

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

    def search_area_rescaling(self):
        if len(self.target_scales) > 0:
            min_scales, max_scales, max_history = 2, 30, 60
            self.target_not_found_counter += 1
            num_scales = max(min_scales, min(max_scales, self.target_not_found_counter))
            target_scales = torch.tensor(self.target_scales)[-max_history:]
            target_scales = target_scales[target_scales >= target_scales[-1]]  # only boxes that are bigger than the `not found`
            target_scales = target_scales[-num_scales:]  # look as many samples into past as not found endures.
            self.target_scale = torch.mean(target_scales) # average bigger boxes from the past

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample.获取提取样本的位置"""
        sample_coord = sample_coord.float()   #sample_coord是坐标
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)  #patches块的中心
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered.获取新样本的中心位置。确保目标正确居中。"""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            train_samples = self.training_samples[0][:self.num_stored_samples[0], ...]   # [2,1024,18,18]
            target_labels = self.target_labels[0][:self.num_stored_samples[0], ...]      # [2,1,18,18]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :]             # [2,4]

            test_feat = self.net.head.extract_head_feat(sample_x)    #提取分类特征 [1,256,18,18]
            train_feat = self.net.head.extract_head_feat(train_samples)     #[2,256,18,18]

            train_ltrb = self.encode_bbox(target_boxes)         #boxes编码
            #解耦cls和bbreg分支
            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc = \
                self.net.head.get_filter_and_features_in_parallel(train_feat, test_feat, num_gth_frames=self.num_gth_frames,
                                                                  train_label=target_labels, train_ltrb_target=train_ltrb)

            # fuse encoder and decoder features to one feature map
            target_scores = self.net.head.classifier(cls_test_feat_enc, cls_weights)


            # compute the final prediction using the output module
            bbox_preds = self.net.head.bb_regressor(bbreg_test_feat_enc, bbreg_weights)

        return target_scores, bbox_preds

    def localize_target(self, scores, sample_pos, sample_scales,center = None):
        """Run the target localization."""

        scores = scores.squeeze(1)

        if center is not None:
            # 生成以lstm预测出来的center值为中心的Gaussian分布
            center = torch.min(center, self.feature_sz)
            sigma_factor = 0.055469755418553784
            sigma = sigma_factor * (self.feature_sz).prod().sqrt().item()
            lstm_score_map = self.gauss_2d(sz=self.feature_sz , sigma=sigma, center=center)  # 基于预测出来的位置的guass
            lstm_score_map = lstm_score_map.to(self.params.device)

            scores = 0.3 * lstm_score_map + 0.7 * scores  # 将lstm score 和 dimp score 加权融合


        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None, max_disp

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative', max_disp2
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1

        return translation_vec1, scale_ind, scores_hn, 'normal', max_disp1

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,   #在某一中心点处，采样一定大小和尺度的图像块
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_backbone_head_feat(self, backbone_feat):
        with torch.no_grad():
            return self.net.get_backbone_head_feat(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples.执行数据扩充以生成初始train样本"""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:   # mode == 'inside_major' 与训练中选的一样
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size  计算增强的尺寸
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations  不同的增强方式
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches  相当于train中self.processing（data）阶段
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)  #[1,3,288,288]

        # Extract initial backbone features  resnet50初始特征提取
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)  #[1,1024,18,18]

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples.获取初始增强样本的目标边界框"""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1,
                                                     x.shape[2] + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] + (int(self.kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('output_sigma_factor', 1/4)
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized img_coords
        target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)

        for target, x in zip(self.target_labels, train_x):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center, end_pad=ksz_even)

        return self.target_labels[0][:train_x[0].shape[0]]

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5*ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_classifier(self, init_backbone_feat):
        # Get classification features 得到分类特征,用resnet layer3的特征作为输入特征
        x = self.get_backbone_head_feat(init_backbone_feat)  #[1,1024,18,18]

        # Add the dropout augmentation here, since it requires extraction of the classification features  在此处添加dropout,因为需要提取分类特征
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):   #不进入
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))   # 特征图大小 [18,18]
        ksz = getattr(self.net.head.filter_predictor, 'filter_size', 1)   #kernel size  1
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz) # [1,1]
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2   #输出的特征大小 [18,18]

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):   #不进入
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations  获取targetboxes不同的数据增强
        target_boxes = self.init_target_boxes()

        # Get target labels for the different augmentations
        self.init_target_labels(TensorList([x]))

        self.num_gth_frames = target_boxes.shape[0]

        if hasattr(self.net.head.filter_predictor, 'num_gth_frames'):
            self.net.head.filter_predictor.num_gth_frames = self.num_gth_frames

        self.init_memory(TensorList([x]))

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
                self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')

#####################################################################################

    def get_samples(self, dataset, in_seq_len, pred_len, device):
        step = 1
        traindata = []
        targetdata = []
        for i in range(0, dataset.shape[0] - in_seq_len, step):
            train_data = dataset[i:i + in_seq_len]
            target_data = dataset[i + in_seq_len:i + in_seq_len + pred_len]
            traindata.append(train_data)
            targetdata.append(target_data)

        traindata = torch.tensor([item.cpu().detach().numpy() for item in traindata], dtype=torch.float).to(device)
        targetdata = torch.tensor([item.cpu().detach().numpy() for item in targetdata], dtype=torch.float).to(device)
        return traindata, targetdata

    def gauss_2d(self, sz, sigma, center, end_pad=(0, 0), density=False):
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        return self.gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1,
                                                                                                -1) * \
               self.gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)

    def gauss_1d(self, sz, sigma, center, end_pad=0, density=False):
        # sz = 18, sigma = 0.9, center, end_pad =（0，0）,
        # k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
        k = torch.arange(0, sz).reshape(1, -1)
        gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
        if density:
            gauss /= math.sqrt(2 * math.pi) * sigma
        return gauss