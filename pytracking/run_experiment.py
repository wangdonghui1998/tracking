import os
import sys
import argparse
import importlib

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.running import run_dataset


def run_experiment(experiment_module: str, experiment_name: str, debug=0, threads=0,visdom_info=None):
    """Run experiment.
    args:
        experiment_module: Name of experiment module in the experiments/ folder.
        experiment_name: Name of the experiment function.
        debug: Debug level.
        threads: Number of threads.
    """
    expr_module = importlib.import_module('pytracking.experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)
    trackers, dataset = expr_func()
    print('Running:  {}  {}'.format(experiment_module, experiment_name))
    run_dataset(dataset, trackers, debug, threads,visdom_info=visdom_info)


def main():
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument('--experiment_module', type=str, default="myexperiments",help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('--experiment_name', type=str,default="otb_test", help='Name of the experiment function.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')    #default = 0没有visdom  default = 1 有序列跟踪时的框 ，default=2 有框有Score map， default=3 有框有Score map有loss
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')

    # parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    # parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    # parser.add_argument('--visdom_port', type=int, default=8098, help='Port for visdom.')

    args = parser.parse_args()

    run_experiment(args.experiment_module, args.experiment_name, args.debug, args.threads)
            # ,{'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port})


if __name__ == '__main__':
    main()
