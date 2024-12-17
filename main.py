import argparse
import yaml
from train import Tracker


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='configs/default.yml', help='Path to the config file')
    parser.add_argument('--dataset', default=None, help='Dataset name')
    parser.add_argument('--data_dir', default=None, help='Path to the data directory')
    parser.add_argument('--network', choices=['unet', 'transformer', 'fc', 'autoencoder', 'cnn'], help='Unet version')
    parser.add_argument('--model_dir', default=None, help='Path to the model directory, to save logs and checkpoints')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--resume', default=False, help='Path to the checkpoint file')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    print('Config:', config)

    tracker = Tracker(config)
    tracker.train()

    tracker.writer.flush()
    tracker.writer.close()


if __name__ == '__main__':
    main()