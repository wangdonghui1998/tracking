class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/home/b311/data/anaconda3/envs/wdh-pytracking/lib/python3.7/site-packages/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/home/b311/data/wdh/pytracking/pytracking/pretrained_networks/'
        #lasot位置可能有问题，先不要用
        self.lasot_dir = '/home/b311/data/wdh/datasets/LaSOT/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/home/b311/data/wdh/datasets/GOT-10k/full_data/train'
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = ''
        self.glass_dir = '/home/b311/data/wdh/datasets/glass/'
        self.samiler_dir = '/home/b311/data/wdh/datasets/Samiler/train'
