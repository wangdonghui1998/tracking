import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features

import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.transformer.transformer as trans
import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads
from lstm_model.lstm_series_model import LSTM,BiLSTM


import matplotlib.pyplot as plt
import visdom
count = 0

class LSToMPNet(nn.Module):
    def train(self, mode=True):
        self.training = mode

        self.backbone_feature_extractor.train(False)
        self.tomp_head.train(False)
        return self

    def __init__(self, backbone_feature_extractor, tomp_head, tomp_head_layer,lstm_net,separate_filters_for_cls_and_bbreg):
        super().__init__()

        self.backbone_feature_extractor = backbone_feature_extractor
        self.tomp_head = tomp_head
        self.tomp_head_layer = [tomp_head_layer] if isinstance(tomp_head_layer, str) else tomp_head_layer
        self.output_layers = sorted(list(set(self.tomp_head_layer)))
        self.lstm_net=lstm_net
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):  # 数据传入model实例化对象后会自动调用
        raise NotImplementedError

    def tomp_train_classifier(self,train_imgs, test_imgs, train_bb,*args, **kwargs):
        # Extract backbone features  resnet 提取主干特征
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))  #[2,1024,18,18]
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))   #[1,1024,18,18]
        # Classification features  获取分类特征（用resnet layer3的特征）
        train_feat_head = self.get_backbone_head_feat(train_feat)  # [2,1024,18,18]
        test_feat_head = self.get_backbone_head_feat(test_feat)  # [1,1024,18,18]

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]
        if train_feat_head.dim() == 5:
            train_feat_head = train_feat_head.reshape(-1, *train_feat_head.shape[-3:])
        if test_feat_head.dim() == 5:
            test_feat_head = test_feat_head.reshape(-1, *test_feat_head.shape[-3:])

        # Extract features 提取分类特征
        train_feat_head = self.extract_head_feat(train_feat_head,num_sequences)  # [2,1,256,18,18]
        test_feat_head = self.extract_head_feat(test_feat_head,num_sequences)    #[1,1,256,18,18]
        # cls_filter[1,256,1,1], breg_filter[1,256,1,1], test_feat_enc[1,1,256,18,18]

        #两种方案——1 不解耦分类和回归分支
        cls_filter, breg_filter, test_feat_enc = self.tomp_head.get_filter_and_features(train_feat_head, test_feat_head, *args, **kwargs)
        # cls_weights[1,256,1,1], bbreg_weights[1,256,1,1], cls_test_feat_enc[1,1,256,18,18]
        return cls_filter, breg_filter,test_feat_enc

        #两种方案——2 解耦分类和回归分支——目前还有问题
        # cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc = \
        #     self.net.tomp_head.get_filter_and_features_in_parallel(train_feat_head, test_feat_head, num_gth_frames=self.num_gth_frames,
        #                                                       train_label=target_labels, train_ltrb_target=train_ltrb)
        # return  cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.backbone_feature_extractor(im, layers)

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.tomp_head_layer})
        #self.head_layer=['layer3'],feat=OrderedDict([('layer3',tensor())])类型参数，shape=[2,1024,18,18]
        if len(self.tomp_head_layer) == 1:
            return feat[self.tomp_head_layer[0]]   #返回的是一个特征图,tensor()
        return feat

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features.  基于输入主干特征提取分类特征"""
        if self.tomp_head.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.tomp_head.feature_extractor(feat)

        output = self.tomp_head.feature_extractor(feat)  # 调用features.residual_bottleneck() 函数  [2,256,18,18]
        '''Sequential(
        (0): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): InstanceL2Norm()
        )
        结果：output.shape = [2,256,18,18]
        '''
        return output.reshape(-1, num_sequences, *output.shape[-3:])  # *output.shape[-3:] 表示后3个维度不变

@model_constructor
def tomp_lstm_net_res50(filter_size=4, tomp_head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, appearance_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True,
              train_feature_extractor=True, train_giounet=True,

             input_size=2,hidden_size=64,num_layers=3,output_size=2):

    # ######################## backbone ########################
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained,frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (appearance_feature_dim * filter_size * filter_size))

    # ######################## classifier ########################
    # Classifier features
    if tomp_head_layer == 'layer3':
        feature_dim = 256
    elif tomp_head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=appearance_feature_dim)

    transformer = trans.Transformer(d_model=appearance_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    # The classifier module
    classifier = heads.LinearFilterClassifier(num_channels=appearance_feature_dim)

    # Bounding box regressor
    bb_regressor = heads.DenseBoxRegressor(num_channels=appearance_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    #初始化 lstm 模块
    # convlstm_net = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
    #                           num_layers=num_layers, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)

    lstm_net = BiLSTM(input_size=input_size,hidden_size=hidden_size,
                    num_layers=num_layers,output_size=output_size)

    #少一个最后的预测器模块response_predictor，然后把response_predictor也传进网络，最后训练response_predictor

    net = LSToMPNet(backbone_feature_extractor=backbone_net, tomp_head=head, tomp_head_layer=tomp_head_layer,
                    lstm_net=lstm_net, separate_filters_for_cls_and_bbreg=True
                    )
    return net
