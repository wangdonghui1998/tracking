import torch.nn as nn
import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq,Samiler
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.kys.utils import DiMPScoreJittering
import ltr.models.tracking.lsdimpnet as lsdimp_models

import ltr.admin.loading as network_loading
import os


def run(settings):
    settings.description = 'Default train settings for LSDiMP with ResNet50 as backbone.'
    settings.batch_size = 64
    settings.num_workers = 1
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    # settings.print_stats = ['Loss/total', 'Loss/iou', 'ClfTrain/clf_ce', 'ClfTrain/test_loss']

    settings.test_sequence_length = 50  # 用于训练seq
    settings.weight_bbeg = 1.0
    settings.weight_dimp_clf = 1.0
    settings.weight_lstm_clf = 1.0
    settings.weight_lstm = 1.0

    # lstm模块
    settings.input_size = 2  # 输入数据维度
    settings.output_size = 2  # 输出数据维度
    settings.hidden_size = 64  # 隐藏层个数
    settings.num_layers = 1  # 几层lstm

    # Train datasets
    # lasot_train = Lasot(settings.env.lasot_dir, split='train')
    # got10k_train = Got10k(settings.env.got10k_dir, split='train')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir)
    samiler_train = Samiler(settings.env.samiler_dir, split='train')

    # Validation datasets
    # got10k_val = Got10k(settings.env.got10k_dir, split='val')
    samiler_val = Samiler(settings.env.samiler_dir, split='val')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params =None
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    data_processing_train = processing.LSDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    data_processing_val = processing.LSDiMPProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=proposal_params,
                                                    label_function_params=label_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    sequence_sample_info = {'num_train_frames': 2, 'num_test_frames': settings.test_sequence_length,
                            'max_train_gap': 30, 'allow_missing_target': True, 'min_fraction_valid_frames': 0.5,
                            'mode': 'Sequence'}

    # Train sampler and loader
    dataset_train = sampler.LSDiMPSampler([samiler_train], [1],
                                       samples_per_epoch=settings.batch_size  , sequence_sample_info=sequence_sample_info,
                                       processing=data_processing_train,
                                       sample_occluded_sequences=True)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # Validation samplers and loaders
    dataset_val = sampler.LSDiMPSampler([samiler_val], [1],
                                        samples_per_epoch=settings.batch_size  , sequence_sample_info=sequence_sample_info,
                                        processing=data_processing_val,
                                        sample_occluded_sequences=True)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    #使用pre-trained dimp网络
    dimp_weights_path = os.path.join(settings.env.pretrained_networks, 'dimp50.pth')  # 路径声明
    base_net, _ = network_loading.load_network(checkpoint=dimp_weights_path)  # 模型加载

    # Create network and actor
    net = lsdimp_models.lsdimpnet_res50(optim_iter=3,detach_length=0,
                            init_gauss_sigma=output_sigma * settings.feature_sz,
                            train_feature_extractor=False, train_iounet=False,
                            input_size=settings.input_size, hidden_size=settings.hidden_size,
                            num_layers=settings.num_layers, output_size=settings.output_size
                            )

    # Move pre-trained dimp weights  移动dimp网络的权重（不知道在干什么）
    net.backbone_feature_extractor.load_state_dict(base_net.feature_extractor.state_dict())
    net.dimp_classifier.load_state_dict(base_net.classifier.state_dict())
    net.bb_regressor.load_state_dict(base_net.bb_regressor.state_dict())

    # To be safe
    for p in net.backbone_feature_extractor.parameters():
        p.requires_grad_(False)
    for p in net.dimp_classifier.parameters():
        p.requires_grad_(False)
    for p in net.bb_regressor.parameters():
        p.requires_grad_(False)

    objective =  {'iou': nn.MSELoss(),
                'dimp_clf': ltr_losses.LBHingev2(threshold=settings.hinge_threshold, return_per_sequence=False),
                'lstm_clf': ltr_losses.LBHingev2(threshold=settings.hinge_threshold, return_per_sequence=False),
                'lstm' : nn.MSELoss(),
                'clf_acc': ltr_losses.TrackingClassificationAccuracy(threshold=0.25)}  #变量声明

    loss_weight = {'iou': settings.weight_bbeg,
                    'dimp_clf': settings.weight_dimp_clf,
                    'lstm_clf':settings.weight_lstm_clf,
                    'lstm':settings.weight_lstm}

    dimp_jitter_fn = DiMPScoreJittering(distractor_ratio=0.1, p_distractor=0.3, max_distractor_enhance_factor=1.3,
                                        min_distractor_enhance_factor=0.8)  # 不知道在干什么，好像是dimp抖动，用来预测score map

    actor = actors.LSDiMPActor(net=net, objective=objective, loss_weight=loss_weight,
                               dimp_jitter_fn=dimp_jitter_fn)

    # Optimizer
    # optimizer = optim.Adam([{'params': actor.net.lstm_net.parameters(), 'lr': 1e-3}])
    optimizer = optim.Adam(actor.net.lstm_net.parameters(), lr=1e-3)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    # trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings)

    trainer.train(500, load_latest=False, fail_safe=True)
