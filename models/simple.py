import torch
import torch.nn as nn

from models.Base import BasePositionPredictor


class FCPositionPredictor(BasePositionPredictor):
    """
    Fully connected position predictor. This is baseline model.

    Input: conditions (batch_size, interval, 8), with 8 stands for (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
    Output: delta_bbox (batch_size, 4)
    """
    def __init__(self, config):
        super(FCPositionPredictor, self).__init__(config)

        self.fc1 = nn.Linear(self.config['interval'] * 8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 4)

    def forward(self, conditions):
        batch_size = conditions.size(0)
        x = conditions.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        delta_bbox = self.fc9(x)
        return delta_bbox
