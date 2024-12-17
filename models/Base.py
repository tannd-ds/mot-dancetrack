import numpy as np
import torch
from torch import nn as nn


class BasePositionPredictor(nn.Module):
    def __init__(self, config):
        super(BasePositionPredictor, self).__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    def generate(self, conditions, img_w, img_h, **kwargs):
        cond_encodeds = []
        for i in range(len(conditions)):
            tmp_c = torch.tensor(np.array(conditions[i]), dtype=torch.float)
            # pad the condition to the interval
            if len(tmp_c) != self.config['interval']:
                pad_conds = tmp_c[-1].repeat((self.config['interval'] - len(tmp_c), 1))
                tmp_c = torch.cat((tmp_c, pad_conds), dim=0)
            cond_encodeds.append(tmp_c.unsqueeze(0))

        cond_encodeds = torch.cat(cond_encodeds)
        cond_encodeds[:, :, 0::2] = cond_encodeds[:, :, 0::2] / img_w
        cond_encodeds[:, :, 1::2] = cond_encodeds[:, :, 1::2] / img_h
        track_pred = self.forward(cond_encodeds.to("cuda"))
        return track_pred.cpu().detach().numpy()
