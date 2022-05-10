import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--dec', default='不加边', type=str)
    parser.add_argument('--dataset', default='NCI1', type=str)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=350, type=int)
    parser.add_argument('--iters_per_epoch', default=50, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--alpha', default='[1, 0]', type=str)
    parser.add_argument('--cleaned', default='False', type=str)
    parser.add_argument('--fold_idx', default='[0]', type=str)
    return parser.parse_args()


args = parse_args()
args.alpha = eval(args.alpha)
args.cleaned = eval(args.cleaned)
if args.fold_idx == 'all':
    args.fold_idx = [i for i in range(10)]
else:
    args.fold_idx = eval(args.fold_idx)
# print(args)
# if __name__ == '__main__':
#     args = parse_args()
#     args.alpha = eval(args.alpha)
#     print(args)
#     print(args.alpha)
