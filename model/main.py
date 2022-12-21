import argparse
from train import trainer
from evaluate import tester

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--output_dir', type=str, default='./checkpoint')

    parser.add_argument('--data_path', type=str, default='./../dataset')

    parser.add_argument('--model', type=str, default='dcgan')

    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if args.do_train == True:
        trainer(args)
    if args.do_eval == True:
        tester(args)