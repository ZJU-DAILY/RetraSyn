import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epsilon', type=float, default=1.0,
                    help='Privacy budget')
parser.add_argument('--grid_num', type=int, default=6,
                    help='Number of grids is n x n')
parser.add_argument('--w', type=int, default=20,
                    help='Window size')
parser.add_argument('--method', type=str, default='retrasyn')
parser.add_argument('--dataset', type=str, default='tdrive')
parser.add_argument('--multiprocessing', action='store_true')
parser.add_argument('--phi', type=int, default=20,
                    help='size of evaluation time range')


args = parser.parse_args()
