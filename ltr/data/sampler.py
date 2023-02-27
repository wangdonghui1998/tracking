import random
import torch.utils.data
from pytracking import TensorDict


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames, used to learn the DiMP classification model and obtain the
    modulation vector for IoU-Net, and ii) a set of test frames on which target classification loss for the predicted
    DiMP model, and the IoU prediction loss for the IoU-Net is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    类描述：生成训练样本，每个训练样本包含[训练帧，测试帧]，
           训练帧用于DiMP分类模型并获得IoU网络的调制向量，
           测试帧用于在这组测试帧上计算预测的DiMP模型的目标分类损失和IoU网络中的IoU预测损失。
    采样方法：随机选择数据集，然后选择该数据集中的一个序列，然后随机采样基本帧（base frame）
             训练帧取样[base_frame_id - max_gap, base_frame_id] 和 测试帧取样[base_frame_id, base_frame_id + max_gap]
             仅对目标可见的帧进行采样，如果没有找到足够的可见帧，则“max_gap”逐渐增加，直到找到足够的帧
             采样帧需要通过'processing' 函数进行处理
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled   包含采样概率的数据集
            samples_per_epoch - Number of training samples per epoch   一个epoch 训练样本数
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.  train frame和test frame之间最大的gap
            num_test_frames - Number of test frames to sample.   要采样的测试帧数
            num_train_frames - Number of train frames to sample.  要采样的训练帧数
            processing - An instance of Processing class which performs the necessary processing of the data.  数据处理
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.  采样模式
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize 取样每个数据集的概率
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch   #每个epoch多少个样本 40000
        self.max_gap = max_gap          #200
        self.num_test_frames = num_test_frames      #1
        self.num_train_frames = num_train_frames    #2
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode      #'causal'

    def __len__(self):   #在执行len()时自动调用
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame   每个帧的目标是否可见
            num_ids - number of frames to be samples        采样帧数
            min_id - Minimum allowed frame number           允许的最小帧号
            max_id - Maximum allowed frame number           允许的最大帧数

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found. 列表，采样帧编号的列表，如果找不到足够可见帧，则为None
        """

        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]       #i（0，len(visible)）获取可见目标的帧的序号 ， visible中存的是0/1

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)   #任选num_ids个可见目标帧

    def __getitem__(self, index): #使用索引访问元素时 或 迭代器enumerate 被自动调用，即在执行[]时自动调用
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames  在序列中取样足够的可见帧
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence 取样一个序列
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)    #train序列总数 GOT-10k:7934     lasot:1120

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)  #调用某一个数据集的get_sequence_info()方法，获取sequence信息,valid类型是bool,visible类型是0/1，
                                                               #got10k {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
                                                               #lasot {'bbox': bbox, 'valid': valid, 'visible': visible}
            visible = seq_info_dict['visible']

            # a = visible.type(torch.int64).sum().item()   #可见目标的数量
            # b = visible.type(torch.int64).sum().item() > 2  #可见目标的数量 >2 吗
            # c = self.num_test_frames + self.num_train_frames   #tomp开头定义的 训练的图片对：num_train_frames=2，测试的图片num_test_frames=1

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames) and len(visible) >= 20    #(可见目标数量>2)*num(train_frame + test_frame) and 序列中可见目标帧数>=20

            enough_visible_frames = enough_visible_frames or not is_video_dataset    #执行完while后enough_visible_frames = true,即现在找到了足够的可见帧

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame  第一帧定义的间隔内的采样帧编号
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_train_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames,
                                                              min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids  以causal方式对框架进行抽样test和train
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible) - self.num_test_frames)   #任选一个可见帧，要求：id(1,总帧数-1)    100-200-0，100

                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])  #取出base帧之前的任意nums_ids（1）个帧，要求：id(base_id - 最大间隔 - 间隔增加，base_id) 即(base_id-200-0,base_id)

                    if prev_frame_ids is None:  #如果取样不到之前的帧，间隔增大
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)     #取出base帧之后的任意1帧，要求：id(base_id+1,base_id + 最大间隔 + 间隔增加) 即(base_id,base_id+200+0)
                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1] * self.num_train_frames
            test_frame_ids = [1] * self.num_test_frames

        train_frames, train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        test_frames, test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name(),
                           'test_class': meta_obj_test.get('object_class_name')})
        return self.processing(data)   #调用processing中的LTRBDenseRegressionProcessing类的__call__函数


class DiMPSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)

class LSDiMPSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False):
        """
        args:
            datasets - List of datasets to be used for training    用于训练的数据集
            p_datasets - List containing the probabilities by which each dataset will be sampled  每个数据集采样的概率的列表
            samples_per_epoch - Number of training samples per epoch  每个epoch的训练样本数
            sequence_sample_info - A dict containing information about how to sample a sequence, e.g. number of frames,
                                    max gap between frames, etc.  包含如何让对序列进行采样的信息，例如帧数，帧间最大间隙
            processing - An instance of Processing class which performs the necessary processing of the data.执行数据的必要处理
            sample_occluded_sequences - If true, sub-sequence containing occlusion is sampled whenever possible. 如果为true，则尽可能对包含遮挡的子序列进行采样
        """

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize 取样每个数据集的概率
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences

    def __len__(self):
        return self.samples_per_epoch

    def _sample_ids(self, visible_ratio, num_ids=1, min_id=None, max_id=None, ratio=0.9):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame  1d张量指示每个帧的目标是否可见
            num_ids - number of frames to be samples    采样帧数
            min_id - Minimum allowed frame number       允许的最小帧号
            max_id - Maximum allowed frame number       允许的最大帧号

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found. 采样帧编号列表。如果找不到足够的可见帧，则无。
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible_ratio):
            max_id = len(visible_ratio)

        visible_ratio_ids = [i for i in range(min_id, max_id) if visible_ratio[i]>ratio]

        # No visible ids
        if len(visible_ratio_ids) == 0:
            return None

        return random.choices(visible_ratio_ids, k=num_ids)

    def _sample_test_ids(self,visible_ratio, num_ids=1, min_id=None, max_id=None):
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible_ratio):
            max_id = len(visible_ratio)

        visible_ratio_ids = [i for i in range(min_id, max_id) if visible_ratio[i]]   #base_id后面的visible_ids

        # No visible ids
        if len(visible_ratio_ids) == 0:
            return None

        sample_list = []
        for i in range (len(visible_ratio_ids)):
            if visible_ratio[visible_ratio_ids[i]]>0 and len(sample_list)<num_ids:
                sample_list.append(visible_ratio_ids[i])
            elif visible_ratio[i]<0:
                continue
            else:
                break
        return sample_list

    def find_occlusion_end_frame(self, first_occ_frame, target_fully_visible):  #返回 下一个目标不完全可见的帧的 下标
        for i in range(first_occ_frame, len(target_fully_visible)):
            if not target_fully_visible[i]:   #target_not_fully_visible : visible_ratio=1 False,visible_ratio<1 True
                return i

        return len(target_fully_visible)   # 如果遍历完还没有，则返回整个序列的长度

    def __getitem__(self, index):    #使用
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks

        修改为：即用前 nums_id帧去预测num_test_frames帧
               随机选取base_id帧
               train_frame: base_id 之前选取 nums_train_frames 个可见帧
               test_frame : base_id 之后选取连续 num_test_frames 个可见帧
        """

        # Select a dataset
        # p_datasets = self.p_datasets
        dataset = random.choices(self.datasets, self.p_datasets)[0]   #任选一个数据集

        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']    #训练的帧数   2
        num_test_frames = self.sequence_sample_info['num_test_frames']      #测试的帧数   30
        max_train_gap = self.sequence_sample_info['max_train_gap']          #取样最大的间隔 30

        # Sample a sequence with enough visible frames  判断序列中是否有满足训练的可见帧
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence 取样一个序列
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)  # train序列总数 GOT-10k:7934     lasot:1120

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)  # 调用某一个数据集的get_sequence_info()方法，获取sequence信息,valid类型是bool,visible类型是0/1，
            # got10k {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
            # lasot {'bbox': bbox, 'valid': valid, 'visible': visible}
            visible = seq_info_dict['visible']
            visible_ratio = seq_info_dict.get('visible_ratio', visible)  # 目标可见比  取值0~1，lasot没有visible_ratio参数，则返回字典中的visible值
            # 可见目标数量 > 2*num(train_frame + test_frame) 且 序列中目标>0.7的个数不小于num_train_frames
            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (num_test_frames + num_train_frames) and\
                                    len(visible_ratio>0.7) > num_train_frames

            enough_visible_frames = enough_visible_frames or not is_video_dataset  # 执行完while后enough_visible_frames = true,即现在找到了足够的可见帧

        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:
                train_frame_ids = None
                test_frame_ids = None
                # gap_increase = 0
                ratio = 0.9

                while test_frame_ids is None:
                    #取样trian帧：2帧，且可见率>0.9  依次递减
                    base_frame_id = self._sample_ids(visible, num_ids=1,min_id=num_train_frames ,max_id=len(visible)-60,ratio=ratio)
                    extra_train_frame_ids = self._sample_ids(visible, num_ids=num_train_frames - 1,
                                                                     min_id=base_frame_id[0] - max_train_gap,
                                                                     max_id=base_frame_id[0],ratio=ratio)
                    if extra_train_frame_ids is None:
                        # gap_increase += 5
                        ratio -= 0.1
                        continue

                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_test_ids(visible_ratio, num_ids=num_test_frames, min_id=base_frame_id[0])

        # Get frames
        train_frames, train_anno_dict, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)   #获取train帧的信息

        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)   #获取test帧的信息
        test_anno = test_anno_dict['bbox']
        # test_valid_anno = test_anno_dict['valid']
        # test_visible = test_anno_dict['visible']
        # test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))
        # test_class = meta_obj_test.get('object_class_name')

        # Prepare data
        data = TensorDict({'train_images': train_frames,     #训练dimp的帧
                           'train_anno': train_anno,        # bbox信息
                           'trian_frame_ids': train_frame_ids,  #train帧的id
                           'test_images': test_frames,      #训练lstm的帧
                           'test_real_images':torch.tensor(test_frames),
                           'test_anno': test_anno,          #bbox信息
                           'test_real_anno':test_anno,
                           'test_frame_ids': test_frame_ids,    #test帧的id
                           'dataset': dataset.get_name()})      #得到dataset的名字

        # Send for processing
        return self.processing(data)

class ATOMSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames=1, num_train_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)


class LWLSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames and ii) a set of test frames. The train frames, along with the
    ground-truth masks, are passed to the few-shot learner to obtain the target model parameters \tau. The test frames
    are used to compute the prediction accuracy.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is randomly
    selected from that dataset. A base frame is then sampled randomly from the sequence. The 'train frames'
    are then sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id], and the 'test frames'
    are sampled from the sequence from the range (base_frame_id, base_frame_id + max_gap] respectively. Only the frames
    in which the target is visible are sampled. If enough visible frames are not found, the 'max_gap' is increased
    gradually until enough frames are found. Both the 'train frames' and the 'test frames' are sorted to preserve the
    temporal order.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, p_reverse=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            p_reverse - Probability that a sequence is temporally reversed
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing

        self.p_reverse = p_reverse

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        visible_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(visible_ids) == 0:
            return None

        return random.choices(visible_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (dataset index)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        reverse_sequence = False
        if self.p_reverse is not None:
            reverse_sequence = random.random() < self.p_reverse

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (self.num_test_frames + self.num_train_frames)

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
            while test_frame_ids is None:
                if gap_increase > 1000:
                    raise Exception('Frame not found')

                if not reverse_sequence:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible)-self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
                else:
                    # Sample in reverse order, i.e. train frames come after the test frames
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_test_frames + 1,
                                                             max_id=len(visible) - self.num_train_frames - 1)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0],
                                                              max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=0,
                                                              max_id=train_frame_ids[0] - 1,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        # Sort frames
        train_frame_ids = sorted(train_frame_ids, reverse=reverse_sequence)
        test_frame_ids = sorted(test_frame_ids, reverse=reverse_sequence)

        all_frame_ids = train_frame_ids + test_frame_ids

        # Load frames
        all_frames, all_anno, meta_obj = dataset.get_frames(seq_id, all_frame_ids, seq_info_dict)

        train_frames = all_frames[:len(train_frame_ids)]
        test_frames = all_frames[len(train_frame_ids):]

        train_anno = {}
        test_anno = {}
        for key, value in all_anno.items():
            train_anno[key] = value[:len(train_frame_ids)]
            test_anno[key] = value[len(train_frame_ids):]

        train_masks = train_anno['mask'] if 'mask' in train_anno else None
        test_masks = test_anno['mask'] if 'mask' in test_anno else None

        data = TensorDict({'train_images': train_frames,
                           'train_masks': train_masks,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_masks': test_masks,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name()})

        return self.processing(data)


class KYSSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False):
        """
        args:
            datasets - List of datasets to be used for training    用于训练的数据集
            p_datasets - List containing the probabilities by which each dataset will be sampled  每个数据集采样的概率的列表
            samples_per_epoch - Number of training samples per epoch  每个epoch的训练样本数
            sequence_sample_info - A dict containing information about how to sample a sequence, e.g. number of frames,
                                    max gap between frames, etc.  包含如何让对序列进行采样的信息，例如帧数，帧间最大间隙
            processing - An instance of Processing class which performs the necessary processing of the data.执行数据的必要处理
            sample_occluded_sequences - If true, sub-sequence containing occlusion is sampled whenever possible. 如果为true，则尽可能对包含遮挡的子序列进行采样
        """

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize 取样每个数据集的概率
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences

    def __len__(self):
        return self.samples_per_epoch

    def _sample_ids(self, valid, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame  1d张量指示每个帧的目标是否可见
            num_ids - number of frames to be samples    采样帧数
            min_id - Minimum allowed frame number       允许的最小帧号
            max_id - Maximum allowed frame number       允许的最大帧号

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found. 采样帧编号列表。如果找不到足够的可见帧，则无。
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(valid):
            max_id = len(valid)

        valid_ids = [i for i in range(min_id, max_id) if valid[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def find_occlusion_end_frame(self, first_occ_frame, target_not_fully_visible):  #返回 下一个目标不完全可见的帧的 下标
        for i in range(first_occ_frame, len(target_not_fully_visible)):
            if not target_not_fully_visible[i]:   #target_not_fully_visible : visible_ratio=1 False,visible_ratio<1 True
                return i

        return len(target_not_fully_visible)   # 如果遍历完还没有，则返回整个序列的长度

    def __getitem__(self, index):    #使用
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        # p_datasets = self.p_datasets
        dataset = random.choices(self.datasets, self.p_datasets)[0]   #任选一个数据集

        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']    #训练的帧数   3
        num_test_frames = self.sequence_sample_info['num_test_frames']      #测试的帧数   50
        max_train_gap = self.sequence_sample_info['max_train_gap']          #取样最大的间隔 30
        allow_missing_target = self.sequence_sample_info['allow_missing_target']  #允许丢失目标 True
        min_fraction_valid_frames = self.sequence_sample_info.get('min_fraction_valid_frames', 0.0)  #有效帧的最小分数 0.5

        if allow_missing_target:
            min_visible_frames = 0  #声明 最小可见帧
        else:
            raise NotImplementedError

        valid_sequence = False     #声明 判断序列是否有效的flag

        # Sample a sequence with enough visible frames and get anno for the same   对足够可见帧的序列进行采样，并获得相同的anno(这是什么)
        while not valid_sequence:

            seq_id = random.randint(0, dataset.get_num_sequences() - 1)   #train序列总数 GOT-10k:7934     lasot:1120

            seq_info_dict = dataset.get_sequence_info(seq_id)  #调用某一个数据集的get_sequence_info()方法，获取sequence信息,
                                                               #got10k {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
                                                               #lasot {'bbox': bbox, 'valid': valid, 'visible': visible}
            visible = seq_info_dict['visible']   #目标可见吗 取值0，1

            visible_ratio = seq_info_dict.get('visible_ratio', visible)     #目标可见比  取值0~1，lasot没有visible_ratio参数，则返回字典中的visible值

            num_visible = visible.type(torch.int64).sum().item()   #可见帧数量

            enough_visible_frames = not is_video_dataset or (num_visible > min_visible_frames and len(visible) >= 20)  # 判断获取的Sampel是否足够

            valid_sequence = enough_visible_frames

        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:
                train_frame_ids = None
                test_frame_ids = None
                gap_increase = 0

                test_valid_image = torch.zeros(num_test_frames, dtype=torch.int8)   #建立测试帧的存储空间
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids 采样开始
                while test_frame_ids is None:
                    #采样目标可见率<0.9的帧（遮挡帧采样）
                    occlusion_sampling = False
                    if dataset.has_occlusion_info() and self.sample_occluded_sequences:  # dataset中是否有遮挡帧True，是否采样遮挡帧True
                        target_not_fully_visible = visible_ratio < 0.9    #取出目标可见率<0.9的
                        if target_not_fully_visible.float().sum() > 0:   #如果目标可见率<0.9的总帧数>0，则完成遮挡帧采样
                            occlusion_sampling = True

                    if occlusion_sampling:    #如果采样目标遮挡帧
                        first_occ_frame = target_not_fully_visible.nonzero()[0]   #第一个遮挡帧，target_not_fully_visible： 1是遮挡帧，0非遮挡帧

                        occ_end_frame = self.find_occlusion_end_frame(first_occ_frame, target_not_fully_visible)  #返回 下一个目标不完全可见的帧的 下标

                        # Make sure target visible in first frame  确保目标在第一帧中完全可见，visible_ratio = 1
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=max(0, first_occ_frame - 20),
                                                         max_id=first_occ_frame - 5)  #随机获取一个base_frame,id:(max（0，first_occ_frame-20）,first_occ_frame-5)

                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)   #取出base帧之前的任意num_ids（3）个帧，要求：id(base_id - 200 - 间隔增加 -1，base_id) 即(base_id-200-1,base_id)

                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids    # base_id之前任选3帧做train帧

                        end_frame = min(occ_end_frame + random.randint(5, 20), len(visible) - 1)    #最后帧：min( 下一个目标不完全可见的帧的 + random(5,20),总帧数-1)

                        if (end_frame - base_frame_id) < num_test_frames:        #最后帧-base_id < 50，即采样肯定不够
                            rem_frames = num_test_frames - (end_frame - base_frame_id)
                            end_frame = random.randint(end_frame, min(len(visible) - 1, end_frame + rem_frames))
                            base_frame_id = max(0, end_frame - num_test_frames + 1)

                            end_frame = min(end_frame, len(visible) - 1)

                        step_len = float(end_frame - base_frame_id) / float(num_test_frames)   #步长 150-101/50

                        test_frame_ids = [base_frame_id + int(x * step_len) for x in range(0, num_test_frames)]   # base_frame 后按步长依次取test帧

                        test_valid_image[:len(test_frame_ids)] = 1   #修改 测试帧 标志

                        test_frame_ids = test_frame_ids + [0] * (num_test_frames - len(test_frame_ids))

                    else: #不必须采样目标遮挡帧
                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=2*num_train_frames,
                                                         max_id=len(visible) - int(num_test_frames * min_fraction_valid_frames))
                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)    # self._sample_ids 返回是列表
                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids    # base_id之前任选3帧做train帧

                        test_frame_ids = list(range(base_frame_id, min(len(visible), base_frame_id + num_test_frames)))  #base_id 后面50帧做测试

                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0]*(num_test_frames - len(test_frame_ids))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Get frames
        train_frames, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)   #获取train帧的信息

        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)   #获取test帧的信息
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        # Prepare data
        data = TensorDict({'train_images': train_frames,    #训练dimp的帧，即得到外观模型的帧
                           'train_anno': train_anno,        #bbox
                           'test_images': test_frames,      #训练kys的帧
                           'test_anno': test_anno,          #bbox
                           'test_valid_anno': test_valid_anno,       #kys的帧是不是可见的（True/False）
                           'test_visible': test_visible,             #kys的帧是不是可见的（0/1）
                           'test_valid_image': test_valid_image,     #
                           'test_visible_ratio': test_visible_ratio, #kys的帧的可见率
                           'dataset': dataset.get_name()})           #dataset的名字

        # Send for processing
        return self.processing(data)

class ToMP_LSTMSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False):
        """
        args:
            datasets - List of datasets to be used for training    用于训练的数据集
            p_datasets - List containing the probabilities by which each dataset will be sampled  每个数据集采样的概率的列表
            samples_per_epoch - Number of training samples per epoch  每个epoch的训练样本数
            sequence_sample_info - A dict containing information about how to sample a sequence, e.g. number of frames,
                                    max gap between frames, etc.  包含如何让对序列进行采样的信息，例如帧数，帧间最大间隙
            processing - An instance of Processing class which performs the necessary processing of the data.执行数据的必要处理
            sample_occluded_sequences - If true, sub-sequence containing occlusion is sampled whenever possible. 如果为true，则尽可能对包含遮挡的子序列进行采样
        """

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize 取样每个数据集的概率
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences

    def __len__(self):
        return self.samples_per_epoch

    def _sample_ids(self, visible_ratio, num_ids=1, min_id=None, max_id=None, ratio=0.9):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame  1d张量指示每个帧的目标是否可见
            num_ids - number of frames to be samples    采样帧数
            min_id - Minimum allowed frame number       允许的最小帧号
            max_id - Maximum allowed frame number       允许的最大帧号

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found. 采样帧编号列表。如果找不到足够的可见帧，则无。
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible_ratio):
            max_id = len(visible_ratio)

        visible_ratio_ids = [i for i in range(min_id, max_id) if visible_ratio[i]>ratio]

        # No visible ids
        if len(visible_ratio_ids) == 0:
            return None

        return random.choices(visible_ratio_ids, k=num_ids)

    def _sample_test_ids(self,visible_ratio, num_ids=1, min_id=None, max_id=None):
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible_ratio):
            max_id = len(visible_ratio)

        visible_ratio_ids = [i for i in range(min_id, max_id) if visible_ratio[i]]   #base_id后面的visible_ids

        # No visible ids
        if len(visible_ratio_ids) == 0:
            return None

        sample_list = []
        for i in range (len(visible_ratio_ids)):
            if visible_ratio[visible_ratio_ids[i]]>0 and len(sample_list)<num_ids:
                sample_list.append(visible_ratio_ids[i])
            elif visible_ratio[i]<0:
                continue
            else:
                break
        return sample_list

    def find_occlusion_end_frame(self, first_occ_frame, target_fully_visible):  #返回 下一个目标不完全可见的帧的 下标
        for i in range(first_occ_frame, len(target_fully_visible)):
            if not target_fully_visible[i]:   #target_not_fully_visible : visible_ratio=1 False,visible_ratio<1 True
                return i

        return len(target_fully_visible)   # 如果遍历完还没有，则返回整个序列的长度

    def __getitem__(self, index):    #使用
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks

        修改为：即用前 nums_id帧去预测num_test_frames帧
               随机选取base_id帧
               train_frame: base_id 之前选取 nums_train_frames 个可见帧
               test_frame : base_id 之后选取连续 num_test_frames 个可见帧
        """

        # Select a dataset
        # p_datasets = self.p_datasets
        dataset = random.choices(self.datasets, self.p_datasets)[0]   #任选一个数据集

        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']    #训练的帧数   2
        num_test_frames = self.sequence_sample_info['num_test_frames']      #测试的帧数   30
        max_train_gap = self.sequence_sample_info['max_train_gap']          #取样最大的间隔 30

        # Sample a sequence with enough visible frames  判断序列中是否有满足训练的可见帧
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence 取样一个序列
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)  # train序列总数 GOT-10k:7934     lasot:1120

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)  # 调用某一个数据集的get_sequence_info()方法，获取sequence信息,valid类型是bool,visible类型是0/1，
            # got10k {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
            # lasot {'bbox': bbox, 'valid': valid, 'visible': visible}
            visible = seq_info_dict['visible']
            visible_ratio = seq_info_dict.get('visible_ratio', visible)  # 目标可见比  取值0~1，lasot没有visible_ratio参数，则返回字典中的visible值
            # 可见目标数量 > 2*num(train_frame + test_frame) 且 序列中目标>0.7的个数不小于num_train_frames
            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (num_test_frames + num_train_frames) and\
                                    len(visible_ratio>0.7) > num_train_frames

            enough_visible_frames = enough_visible_frames or not is_video_dataset  # 执行完while后enough_visible_frames = true,即现在找到了足够的可见帧

        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:
                train_frame_ids = None
                test_frame_ids = None
                # gap_increase = 0
                ratio = 0.9

                while test_frame_ids is None:
                    #取样trian帧：2帧，且可见率>0.9  依次递减
                    base_frame_id = self._sample_ids(visible, num_ids=1,min_id=num_train_frames ,ratio=ratio)
                    extra_train_frame_ids = self._sample_ids(visible, num_ids=num_train_frames - 1,
                                                                     min_id=base_frame_id[0] - max_train_gap,
                                                                     max_id=base_frame_id[0],ratio=ratio)
                    if extra_train_frame_ids is None:
                        # gap_increase += 5
                        ratio -= 0.1
                        continue

                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_test_ids(visible_ratio, num_ids=num_test_frames, min_id=base_frame_id[0])

        # Get frames
        train_frames, train_anno_dict, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)   #获取train帧的信息

        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)   #获取test帧的信息
        test_anno = test_anno_dict['bbox']
        # test_valid_anno = test_anno_dict['valid']
        # test_visible = test_anno_dict['visible']
        # test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))
        # test_class = meta_obj_test.get('object_class_name')

        # Prepare data
        data = TensorDict({'train_images': train_frames,     #训练tomp的帧

                           'train_anno': train_anno,        # bbox信息
                           'trian_frame_ids': train_frame_ids,  #train帧的id
                           'test_images': test_frames,      #训练lstm的帧
                           'test_real_images': torch.tensor(test_frames),
                           'test_anno': test_anno,          #bbox信息
                           'test_real_anno': test_anno,
                           'test_frame_ids': test_frame_ids,    #test帧的id
                           # 'test_valid_anno': test_valid_anno,      #目标是否可见(True/False)
                           # 'test_visible': test_visible,            #目标是否可见(0/1)
                           # 'test_valid_image': test_valid_image,    #
                           # 'test_visible_ratio': test_visible_ratio,    #目标可见率
                           # 'test_class': test_class,            #类别
                           'dataset': dataset.get_name()})      #得到dataset的名字

        # Send for processing
        return self.processing(data)



class SequentialTargetCandidateMatchingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, samples_per_epoch, sup_modes, p_sup_modes=None, processing=no_processing,
                 subseq_modes=None, p_subseq_modes=None, frame_modes=None, p_frame_modes=None):
        """
        args:
            datasets - List of datasets to be used for training
            samples_per_epoch - Number of training samples per epoch
            sup_modes - List of different supervision modes to use (partial_sup or self_sup).
            p_sup_modes - List of sup_mode sample probabilities.
            processing - An instance of Processing class which performs the necessary processing of the data.
            subseq_modes - List of different subsequence modes to sample from (HH, HK, HG), see KeepTrack paper for details.
            p_subseq_modes - List of subseq_mode sample probabilities.
            frame_modes - List of different frame mode to sample from (H, K, J), see KeepTrack paper for details.
            p_frame_modes - List of frame_mode sample probabilities.

        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.processing = processing
        self.subseq_modes = subseq_modes
        self.frame_modes = frame_modes
        self.sup_modes = sup_modes if sup_modes is not None else ['self_sup']
        self.p_sup_modes = p_sup_modes

        if p_sup_modes is None:
            self.p_sup_modes = [1. / len(self.sup_modes)] * (len(self.sup_modes))

        if subseq_modes is not None:
            self.dataset_subseq_states = self._load_dataset_subsequence_states()

            if p_subseq_modes is None:
                p_subseq_modes = [self.dataset_subseq_states[mode].shape[0] for mode in self.subseq_modes]

            # Normalize
            p_subseq_total = sum(p_subseq_modes)
            self.p_subseq_modes = [x / p_subseq_total for x in p_subseq_modes]

        if frame_modes is not None:
            self.dataset_frame_states = self._load_dataset_frame_states()

            if p_frame_modes is None:
                p_frame_modes = [self.dataset_frame_states[mode].shape[0] for mode in self.frame_modes]

            # Normalize
            p_frames_total = sum(p_frame_modes)
            self.p_frame_modes = [x / p_frames_total for x in p_frame_modes]

    def __len__(self):
        return self.samples_per_epoch

    def _load_dataset_subsequence_states(self):
        return self.dataset.get_subseq_states()

    def _load_dataset_frame_states(self):
        return self.dataset.get_frame_states()

    def _sample_valid_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which dumped data is useful

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be sampled
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 2:
            min_id = 2
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        num_begin = num_ids//2
        num_end = num_ids - num_ids//2
        ids_begin = random.sample(valid_ids[:len(valid_ids)//2], k=num_begin)
        ids_end = random.sample(valid_ids[len(valid_ids)//2:], k=num_end)
        return ids_begin + ids_end


    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly).

        returns:
            TensorDict - dict containing all the data blocks
        """

        # select a subseq mode
        sup_mode = random.choices(self.sup_modes, self.p_sup_modes, k=1)[0]

        if sup_mode == 'self_sup':
            mode = random.choices(self.frame_modes, self.p_frame_modes, k=1)[0]

            states = self.dataset_frame_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id]

        elif sup_mode == 'partial_sup':
            mode = random.choices(self.subseq_modes, self.p_subseq_modes, k=1)[0]

            states = self.dataset_subseq_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id, baseframe_id + 1]
        else:
            raise ValueError('Supervision mode: \'{}\' is invalid.'.format(sup_mode))


        seq_info_dict = self.dataset.get_sequence_info(seq_id)

        frames_dict, _ = self.dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({
            'dataset': self.dataset.get_name(),
            'mode': mode,
            'seq_name': self.dataset.get_sequence_name(seq_id),
            'base_frame_id': baseframe_id,
            'sup_mode': sup_mode
        })

        for key, val in frames_dict.items():
            data[key] = val

        return self.processing(data)
