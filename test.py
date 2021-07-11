import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('x', default=10, type=int, nargs=2, help='x')
parser.add_argument('--y', type=int, help='y', default=[])
#print(parser.parse_args('--x 19'.split()))
args = parser.parse_args()
print(args.__dir__)
print(args.x)
print(args.y)
print(type(args.x))