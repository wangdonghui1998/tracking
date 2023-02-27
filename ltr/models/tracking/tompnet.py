import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.transformer.transformer as trans
import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads

import matplotlib.pyplot as plt
import visdom
count = 0
class ToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))


    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):   #数据传入model实例化对象后会自动调用
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).  [2,1,3,288,288]
            test_imgs:  Test image samples (images, sequences, 3, H, W).    [1,1,3,288,288]
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).  [2,1,4]
            *args, **kwargs:  These are passed to the classifier module.  train_label=data['train_label'] [2,1,18,18] ,  train_ltrb_target=data['train_ltrb_target'] [2,1,4,18,18] 也传进来了
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        # vis = visdom.Visdom(port=8098)
        # for i in range(0,train_imgs.shape[0]):
        #     global count
        #     plt.title("train"+str(count)+"-"+str(i))
        #     img = train_imgs[i][0].squeeze()
        #     img = img.swapaxes(0, 1)
        #     img = img.swapaxes(1, 2)
        #     plt.imshow(img.cpu(), cmap='viridis')
        #     vis.matplot(plt)
        #
        # img = test_imgs[0][0].squeeze()
        # img = img.swapaxes(0, 1)
        # img = img.swapaxes(1, 2)
        # plt.imshow(img.cpu(), cmap='viridis')
        # plt.title("test" + str(count))
        # vis.matplot(plt)
        # count = count + 1

        # Extract backbone features  提取主干特征
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
        # self.train_feat['layer3'].shape = [2,1024,18,18]
        # self.test_feat['layer3'].shape = [1,1024,18,18]

        # Classification features  获取分类特征（用resnet layer3的特征）
        train_feat_head = self.get_backbone_head_feat(train_feat)    #[2,1024,18,18]
        test_feat_head = self.get_backbone_head_feat(test_feat)      #[1,1024,18,18]

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)

        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        #self.head_layer=['layer3'],feat=OrderedDict([('layer3',tensor())])类型参数，shape=[2,1024,18,18]
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]   #返回的是一个特征图,tensor()
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor    #网络模型构建
def tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    #特征提取网络
    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    #transform 初始化 encoder--decoder
    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    #位置编码 和 bounding box编码
    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    #线性分类器
    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    #密集边界框回归
    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    #把以上这些东西集成到一个head中
    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)  #仅声明
    return net


@model_constructor
def tompnet101(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
               final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet101(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net
