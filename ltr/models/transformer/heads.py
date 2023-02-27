import torch
import torch.nn as nn
import ltr.models.layers.filter as filter_layer


def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers


class Head(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """
        :param train_feat: [1,1024,18,18]
        :param test_feat: [2,1024,18,18]
        :param train_bb: [2,1,4]            train_bb.dim()=3
        :param args:
        :param kwargs:
        :return:
        """
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]     #1

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features 提取分类特征
        train_feat = self.extract_head_feat(train_feat, num_sequences)     #[2,1,256,18,18]
        test_feat = self.extract_head_feat(test_feat, num_sequences)       #[1,1,256,18,18]

        # Train filter 得到分类和回归权重和增强测试帧   *args 和 **kwargs ：data['train_label'], data['train_ltrb_target']
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)  #"*args 和 **kwargs 主要用于函数定义， 可以将不定数量的参数传递给一个函数。"。
        #cls_filter.shape=[1,256,1,1], breg_filter.shape=[1,256,1,1], test_feat_enc.shape=[1,1,256,18,18]
        # fuse encoder and decoder features to one feature map 目标定位
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features.  基于输入主干特征提取分类特征"""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)   #调用features.residual_bottleneck() 函数  [2,256,18,18]
        '''Sequential(
        (0): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): InstanceL2Norm()
        )
        结果：output.shape = [2,256,18,18]
        '''
        return output.reshape(-1, num_sequences, *output.shape[-3:])  # *output.shape[-3:] 表示后3个维度不变

    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        '''获得分类和回归的权重和增强测试帧test_feat_enc
        feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        :param train_feat: [2,1,256,18,18]
        :param test_feat:  [1,1,256,18,18]
        :param train_label: [2,1,18,18]
        :return:
        '''

        if self.separate_filters_for_cls_and_bbreg:   #true:将分类和回归分支分开
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)  #调用filter_predictor的forward函数
            cls_weights = bbreg_weights = weights   #分类和回归的权重

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)


class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))   #extend()拓展列表，可以把新的列表添加到你列表的末尾
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0) # (nf*ns, 4, h, w)
        return ltrb
