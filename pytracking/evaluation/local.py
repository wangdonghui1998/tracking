from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/home/b311/data/wdh/datasets/GOT-10k/full_data'
    settings.similar_path = '/home/b311/data/wdh/datasets/Similar'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_path = '/home/b311/data/wdh/datasets/LaSOT/LaSOT/LaSOTBenchmark'
    settings.network_path = '/home/b311/data/wdh/pytracking/pytracking/pretrained_networks/'    # Where tracking networks are stored.
    # settings.network_path = '/home/b311/data/wdh/pytracking/ltr/checkpoints/ltr/lsdimp/lsdimp50/'
    settings.nfs_path = ''
    settings.otb_path = '/home/b311/data/wdh/datasets/OTB100/OTB100/'
    settings.oxuva_path = ''
    settings.result_plot_path = '/home/b311/data/wdh/pytracking/pytracking/result_plots/'
    settings.results_path = '/home/b311/data/wdh/pytracking/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/b311/data/wdh/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = '/home/b311/data/wdh/datasets/VOT2017and2018/'
    settings.youtubevos_dir = ''
    settings.glass_path = '/home/b311/data/wdh/datasets/glass/'

    return settings

