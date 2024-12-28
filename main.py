import argparse
import yaml
import time
from train import Tracker


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='configs/default.yml', help='Path to the config file')
    parser.add_argument('--dataset', default=None, help='Dataset name')
    parser.add_argument('--data_dir', default=None, help='Path to the data directory')
    parser.add_argument('--network', choices=['unet', 'transformer', 'fc', 'autoencoder', 'vae', 'cnn', 'tcn'], help='Unet version')
    parser.add_argument('--model_dir', default=None, help='Path to the model directory, to save logs and checkpoints')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--resume', default=False, help='Path to the checkpoint file')
    parser.add_argument('--eval', type=str, help='Evaluate the model')
    args, arbitrary_args = parser.parse_known_args()

    def is_valid_arbitrary_arg_pair(arg1, arg2):
        return arg1.startswith('--') and not arg2.startswith('--')

    # Parse arbitrary arguments
    for i in range(0, len(arbitrary_args), 2):
        k, v = arbitrary_args[i], arbitrary_args[i+1]
        if is_valid_arbitrary_arg_pair(k, v):
            print('Setting', k, v)
            if v == 'None':
                v = None
            if v == 'True':
                v = True
            if v == 'False':
                v = False
            if isinstance(v, str) and v.isdigit():
                v = int(v)
            setattr(args, k[2:], v)
        else:
            print('Invalid argument:', k, v)
    return args


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        if v is not None:
            if v == 'True':
                v = True
            if v == 'False':
                v = False
            config[k] = v

    config['timestamp'] = time.strftime('%Y%m%d-%H%M%S')

    print('Config:', config)

    tracker = Tracker(config)
    if config['eval']:
        tracker.eval()
    else:
        tracker.train()

        tracker.writer.flush()
        tracker.writer.close()


if __name__ == '__main__':
    main()