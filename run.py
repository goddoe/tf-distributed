from os.path import join, abspath, dirname
import argparse

from model import model_f
from tf_distributed import run_train

# ======================================
# Path
PARAM_PATH = join(abspath(dirname(__file__)), "param.json")


# ======================================
# Main
def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='TF server and model info.')
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs")

    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs")

    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'")

    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job")

    parser.add_argument(
        '--dataset',
        metavar='dataset',
        type=str,
        help='path of train csv file')

    parser.add_argument(
        '--output',
        metavar='output',
        type=str,
        help='path of output')

    parser.add_argument(
        "--param_path",
        type=str,
        default=PARAM_PATH,
        help="Parameters of model")

    p = parser.parse_args()

    # Train
    print("{}\ntrain start\n{}".format('*'*30, '*'*30))
    run_train(ps_hosts=p.ps_hosts,
              worker_hosts=p.worker_hosts,
              job_name=p.job_name,
              task_index=p.task_index,
              model_f=model_f,
              data_path=p.dataset,
              output_path=p.output,
              param_path=p.param_path)


if __name__ == '__main__':
    main()
