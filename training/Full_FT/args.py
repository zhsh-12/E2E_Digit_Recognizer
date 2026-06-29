import argparse

def args_parse(args_list=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_ratio', type=str, default='1:1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--mode', type=str, default='FineTuning')

    return parser.parse_args(args_list)