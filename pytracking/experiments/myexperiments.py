from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def similar_test():
    trackers =  trackerlist('lsdimp', 'lsdimp', range(1))
    # trackers = trackerlist('dimp', 'dimp50', range(1))
    # trackers = trackerlist('tomp', 'tomp50', range(1)) + \
    #             trackerlist('lstomp', 'lstomp', range(1))
    # trackers = trackerlist('lsatom', 'lsatom', range(1))
    # trackers = trackerlist('atom', 'default', range(1))
    # trackers = trackerlist('lwl', 'lwl_ytvos', range(1))
    # trackers = trackerlist('dimp', 'prdimp50', range(1))
    # trackers = trackerlist('lsdimp', 'lsprdimp50', range(1))
    # trackers = trackerlist('lsdimp', 'lsprdimp18', range(1))
    dataset = get_dataset('similar_test')
    return trackers, dataset

def otb_test():
    trackers =  trackerlist('lsdimp' ,'lsdimp', range(1))
    # trackers = trackerlist('dimp', 'dimp50', range(1))
    dataset = get_dataset('otb')
    return trackers, dataset

def glass_test():
    trackers = trackerlist('dimp', 'dimp50', range(3))
    dataset = get_dataset('glass')
    return trackers, dataset
