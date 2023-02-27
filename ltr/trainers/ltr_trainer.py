import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import torch
import torch.nn as nn
import time
import warnings
warnings.filterwarnings("ignore")


def freeze_batchnorm_layers(net):
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, freeze_backbone_bn_layers=False):
        """
        args:
            actor - The actor for training the network  训练网络的
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.  数据集加载器列表 [train_loader, val_loader] ，每个epoch trainer 都会运行一个
            optimizer - The optimizer used for training, e.g. Adam  优化器 Adam
            settings - Training settings 训练设置
            lr_scheduler - Learning rate scheduler  学习率调整
            freeze_backbone_bn_layers - Set to True to freeze the bach norm statistics in the backbone during training.
                                        设置为True可在训练期间冻结主干中的BN层
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables  初始化变量的统计信息
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})


        # Initialize tensorboard   初始化tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])
        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

        self.freeze_backbone_bn_layers = freeze_backbone_bn_layers

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation. 进行一个训练或验证周期"""
        self.actor.train(loader.training)   # loader.training == true，此处的actor是ltr.actors.tracking.ToMPActor,如果开启是别的跟踪器，则是别的类
                                            # ToMPActor 继承 BaseActor,train()是BaseActor中的方法
        if self.freeze_backbone_bn_layers:
            freeze_batchnorm_layers(self.actor.net.feature_extractor)

        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):  # 声明采样方法  loader里面有datasets对象的初始化参数，如‘frame_sample_mode','max_gap','num_test_frames','num_train_frames'
                                              # 此时会调用sample中的 __getitem__方法
                                              # loader : 数据集对象 2个

            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            loss, stats = self.actor(data,self.optimizer,loader)   #更新loss和状态，此处调用tracking.ToMPActor.__call__()
            # backward pass and update weights
            # if loader.training:
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

            # update statistics
            self._update_stats(stats, loader.batch_size, loader)
            # print statistics
            self._print_stats(i, loader, loader.batch_size)

    def train_epoch(self):
        """Do one epoch for each loader. 对每个loader做一个epoch"""

        for loader in self.loaders:     #self.loader = 2 猜测因为有两个train数据集 ，loader第一个lasot_train，第二个got10k_train
            if self.epoch % loader.epoch_interval == 0:   #epoch_insterval = 1  纪元间隔
                self.cycle_dataset(loader)

        # self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])


    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)