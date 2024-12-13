import argparse
import yaml
from tqdm import tqdm
from dataset.dataset import DiffMOTDataset, custom_collate_fn

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='', help='Path to the config file')
    parser.add_argument('--dataset', default='', help='Dataset name')
    parser.add_argument('--network', choices=['ReUNet', 'ReUNet+++', 'Smaller'], help='Unet version')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the network')
    parser.add_argument('--filters', type=int, nargs='+', help="List of filters")
    parser.add_argument('--skip_connection', type=str2bool, default=False, help='Skp connection')
    parser.add_argument('--data_dir', default=None, help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--early_stopping', choices=['loss', 'iou'])
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset

    agent = None

    if config.eval_mode:
        agent.eval()
    else:
        agent.train()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class SimpleMOTModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4):
        super(SimpleMOTModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # 3 is from `condition` (interval-1)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Predict delta_bbox (4 values)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)


def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            conditions = batch['condition'].float()  # Input: (B, 3, 8)
            target = batch['delta_bbox'].float()  # Output: (B, 4)

            optimizer.zero_grad()
            output = model(conditions)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")



if __name__ == '__main__':
    # main()

    # Prepare dataset and dataloader
    dataset = DiffMOTDataset("/home/tanndds/my/datasets/dancetrack/trackers_gt_t/train")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    # Model, loss, and optimizer
    model = SimpleMOTModel(input_dim=8, hidden_dim=64, output_dim=4)
    criterion = nn.MSELoss()  # Regression loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)


