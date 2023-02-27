import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
import ltr.models.loss as ltr_losses
import ltr.models.tracking.lstompnet as tomp_lstm_models
import torch.nn as nn
import ltr.actors.tracking as actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr.models.loss.bbr_loss import GIoULoss

import ltr.admin.loading as network_loading
import os


def run(settings):
    settings.description = 'ToMP_LSTM50'
    settings.batch_size = 1
    settings.num_workers = 1
    settings.multi_gpu = False

    settings.test_sequence_length = 30 # 用于训练seq
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 1
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3.0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.3}
    settings.hinge_threshold = 0.05
    settings.num_train_frames = 2
    settings.num_test_frames = 1
    settings.num_encoder_layers = 6
    settings.num_decoder_layers = 6
    settings.frozen_backbone_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    settings.freeze_backbone_bn_layers = True

    settings.crop_type = 'inside_major'
    settings.max_scale_change = 1.5
    settings.max_gap = 200
    settings.train_samples_per_epoch = 40000
    settings.val_samples_per_epoch = 10000
    settings.val_epoch_interval = 5
    settings.num_epochs = 300

    settings.weight_bbeg = 1.0
    settings.weight_test_clf = 1000.0
    settings.weight_lstm = 1.0
    settings.normalized_bbreg_coords = True
    settings.center_sampling_radius = 1.0
    settings.use_test_frame_encoding = False  # Set to True to use the same as in the paper but is less stable to train.

    #lstm模块
    settings.input_size = 2     #输入数据维度
    settings.output_size = 2    #输出数据维度
    settings.hidden_size = 64    #隐藏层个数
    settings.num_layers = 3     #几层lstm

    settings.kernel_size = (3, 3)   #convlstm
    settings.convlstm_clf = 1



    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')

    # got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    got10k_train = Got10k(settings.env.got10k_dir, split='train')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    # got10k_val = Got10k(settings.env.got10k_dir, split='votval')
    got10k_val = Got10k(settings.env.got10k_dir, split='val')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),    #以概率转为灰度图像
                                    tfm.RandomHorizontalFlip(probability=0.5))  #以概率水平反转图像

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),  #以概率转换为张量和抖动亮度
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)) #归一化图像

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module  跟踪对处理模块
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    #用于边界盒回归,ltrb编码
    data_processing_train = processing.LSTOMPProcessing(search_area_factor=settings.search_area_factor,
                                                                     output_sz=settings.output_sz,
                                                                     center_jitter_factor=settings.center_jitter_factor,
                                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                                     crop_type=settings.crop_type,
                                                                     max_scale_change=settings.max_scale_change,
                                                                     mode='sequence',
                                                                     label_function_params=label_params,
                                                                     transform=transform_train,
                                                                     joint_transform=transform_joint,
                                                                     use_normalized_coords=settings.normalized_bbreg_coords,
                                                                     center_sampling_radius=settings.center_sampling_radius)

    data_processing_val = processing.LSTOMPProcessing(search_area_factor=settings.search_area_factor,
                                                                   output_sz=settings.output_sz,
                                                                   center_jitter_factor=settings.center_jitter_factor,
                                                                   scale_jitter_factor=settings.scale_jitter_factor,
                                                                   crop_type=settings.crop_type,
                                                                   max_scale_change=settings.max_scale_change,
                                                                   mode='sequence',
                                                                   label_function_params=label_params,
                                                                   transform=transform_val,    #此处不同
                                                                   joint_transform=transform_joint,
                                                                   use_normalized_coords=settings.normalized_bbreg_coords,
                                                                   center_sampling_radius=settings.center_sampling_radius)

    # Train sampler and loader
    #,trackingnet_train, coco_train  , 1, 1
    sequence_sample_info = {'num_train_frames': 2, 'num_test_frames': settings.test_sequence_length,
                            'max_train_gap': 30, 'allow_missing_target': True, 'min_fraction_valid_frames': 0.5,
                            'mode': 'Sequence'}

    dataset_train = sampler.ToMP_LSTMSampler([got10k_train, lasot_train], [0.3, 0.25],   #0.25
                                       samples_per_epoch=settings.batch_size * 150,
                                       sequence_sample_info=sequence_sample_info,
                                       processing=data_processing_train,
                                       sample_occluded_sequences=True)


    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    dataset_val = sampler.ToMP_LSTMSampler([got10k_val], [1], samples_per_epoch=1000,
                                           sequence_sample_info=sequence_sample_info, processing=data_processing_val,
                                           sample_occluded_sequences=True)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=settings.val_epoch_interval, stack_dim=1)

    #使用pre-trained tomp网络
    tomp_weights_path = os.path.join(settings.env.pretrained_networks, 'tomp50.pth.tar')  # 路径声明
    base_net, _ = network_loading.load_network(checkpoint=tomp_weights_path)      # 模型加载


    net = tomp_lstm_models.tomp_lstm_net_res50(filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
                            head_feat_norm=True, final_conv=True, appearance_feature_dim=256, feature_sz=settings.feature_sz,
                            frozen_backbone_layers=settings.frozen_backbone_layers,
                            num_encoder_layers=settings.num_encoder_layers,
                            num_decoder_layers=settings.num_decoder_layers,
                            use_test_frame_encoding=settings.use_test_frame_encoding,
                            train_feature_extractor=False, train_giounet=False,

                            input_size=settings.input_size, hidden_size=settings.hidden_size,
                            num_layers=settings.num_layers, output_size=settings.output_size)    #初始化已没问题
    #
    # Move pre-trained tomp weights  移动tomp网络的权重（不知道在干什么）
    net.backbone_feature_extractor.load_state_dict(base_net.feature_extractor.state_dict())
    net.tomp_head.load_state_dict(base_net.head.state_dict(),False)


    # To be safe
    for p in net.backbone_feature_extractor.parameters():
        p.requires_grad_(False)

    objective = {'tomp_giou': GIoULoss(),
                 'test_clf': ltr_losses.LBHingev2(threshold=settings.hinge_threshold),   # 变量声明
                 'lstm_clf': ltr_losses.LBHingev2(threshold=settings.hinge_threshold),
                 'clf_acc': ltr_losses.TrackingClassificationAccuracy(threshold=0.25)}

    loss_weight = {'tomp_giou': settings.weight_bbeg, 'test_clf': settings.weight_test_clf, 'lstm':settings.weight_lstm}  # 变量声明

    actor = actors.LSToMPActor(net=net, objective=objective, loss_weight=loss_weight)  # 仅声明

    optimizer = optim.AdamW([
        {'params': actor.net.tomp_head.parameters(), 'lr': 1e-4},
        {'params': actor.net.backbone_feature_extractor.layer3.parameters(), 'lr': 2e-5},
        {'params': actor.net.lstm_net.parameters(),'lr': 1e-3},# 变量声明
    ], lr=2e-4, weight_decay=0.0001)   #2e-4

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.2)
    #
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)  # 仅声明
    # # LTRTrainer 继承了 BaseTrainer ，train()中的train_epoch()函数是LTRTrainer重写的函数
    trainer.train(10, load_latest=True, fail_safe=True)  # 开始调用