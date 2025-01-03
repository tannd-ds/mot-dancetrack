import os
import yaml
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import TrackingDataset, custom_collate_fn, augment_data
from utils import calculate_iou, calculate_ade, original_shape
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from models.TransformerBase import *
from models.Autoencoder import *
from models.Convolution import *
from models.simple import *

class Tracker(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0

        self._init_model()
        self._init_model_dir()
        self._init_data_loader()
        self._init_optimizer()
        self._init_tensorboard()

        os.makedirs(os.path.join(self.config['model_dir'], 'weights'), exist_ok=True)


    def train(self):
        """ Train the model """
        torch.backends.cudnn.benchmark = True
        k_fold = KFold(n_splits=5, shuffle=True)

        print("Training the model...")
        for epoch in range(self.config['epochs']):
            train_indices, val_indices = next(k_fold.split(self.train_data))
            train_set = torch.utils.data.Subset(self.train_data, train_indices)
            val_set = torch.utils.data.Subset(self.train_data, val_indices)

            train_set_loader = DataLoader(
                train_set,
                batch_size=self.config['batch_size'],
                shuffle=True,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True,
            )
            val_set_loader = DataLoader(
                val_set,
                batch_size=self.config['batch_size'],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True,
            )

            print('\033[92m', end='') # green
            self.step(train_set_loader, train=True)
            print('\033[93m', end='') # yellow
            self.step(val_set_loader, train=False, log_writer=False)
            print('\033[0m', end='') # white

            if (epoch + 1) % self.config['eval_every'] == 0:
                self.step(self.val_data_loader, train=False)

                save_dir = os.path.join(self.config['model_dir'], 'weights', f"epoch_{self.epoch}.pt")
                torch.save(self.model.state_dict(), save_dir)
                print(f"Model saved at {save_dir}")

            self.epoch += 1


    def step(self, data_loader, train=True, log_writer=True):
        self.model.train() if train else self.model.eval()
        epoch_loss = 0

        total_iou = 0
        total_ade = 0

        for batch in tqdm(data_loader, desc=f"Epoch {self.epoch}/{self.config['epochs']}"):
            for key in batch:
                batch[key] = batch[key].to(self.device)

            conditions = augment_data(batch['condition'].float())
            delta_bbox = batch['delta_bbox'].float()

            with torch.amp.autocast('cuda'):
                if self.config['network'] == 'autoencoder':
                    recon, predicted_delta_bbox = self.model(conditions)
                    recon = recon.view(-1, 9, 8)
                    recon_loss = self.criterion(recon, conditions)
                    delta_loss = self.criterion(predicted_delta_bbox, delta_bbox)
                    loss = recon_loss + delta_loss
                elif self.config['network'] == 'vae':
                    recon, predicted_delta_bbox, mu, logvar = self.model(conditions)
                    loss = self.model.loss_function(recon, conditions, mu, logvar)
                else:
                    predicted_delta_bbox = self.model(conditions)
                    loss = self.criterion(predicted_delta_bbox, delta_bbox)

            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # MeanIoU and MeanADE
            prev_bbox = conditions[:, -1, :4]
            pred_bbox = prev_bbox + predicted_delta_bbox
            target_bbox = batch['cur_bbox']
            w, h = batch['width'], batch['height']

            orig_pred = original_shape(pred_bbox, w, h)
            orig_target = original_shape(target_bbox, w, h)

            # Calculate IoU
            total_iou += calculate_iou(orig_pred, orig_target)
            total_ade += calculate_ade(orig_pred, orig_target)

            epoch_loss += loss.item()

        print(f"Epoch [{self.epoch}/{self.config['epochs']}],",
              f"Loss: {epoch_loss / len(data_loader):.8f},",
              f"MeanIoU: {total_iou / len(data_loader):.8f},",
              f"MeanADE: {total_ade / len(data_loader):.8f}")

        if log_writer:
            if train:
                self.writer.add_scalar("Loss/train", epoch_loss / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanIoU/train", total_iou / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanADE/train", total_ade / len(data_loader), self.epoch)
            else:
                self.writer.add_scalar("Loss/val", epoch_loss / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanIoU/val", total_iou / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanADE/val", total_ade / len(data_loader), self.epoch)

        self.scheduler.step()


    def _init_data_loader(self):
        train_path = os.path.join(self.config['data_dir'], 'train')
        val_path = os.path.join(self.config['data_dir'], 'val')
        train_dataset = TrackingDataset(train_path, config=self.config)
        val_dataset = TrackingDataset(val_path, config=self.config)
        print(f"Number of samples in the Train dataset: {len(train_dataset)}")
        print(f"Number of samples in the validation dataset: {len(val_dataset)}")

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.val_data_loader = val_data_loader


    def _init_model(self):
        if self.config['network'] == 'transformer':
            model = TransformerPositionPredictor(self.config, emb_dim=8)
        elif self.config['network'] == 'fc':
            model = FCPositionPredictor(self.config)
        elif self.config['network'] == 'autoencoder':
            model = AutoEncoderPositionPredictor(self.config)
        elif self.config['network'] == 'cnn':
            model = Conv2dPredictor(self.config)
        elif self.config['network'] == 'vae':
            model = VAEPositionPredictor(self.config)

        if self.config['resume']:
            if not os.path.exists(self.config['resume']):
                print('Checkpoint file not found, training from scratch...')
            else:
                # get the number of epochs from the checkpoint file
                self.epoch = int(self.config['resume'].split('/')[-1].split('.')[0].split('_')[-1]) + 1
                model.load_state_dict(torch.load(self.config['resume']))
                print('Model loaded from ', self.config['resume'])

        print('Number of Model\'s parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.model = model.to(self.device)


    def _init_optimizer(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        self.scaler = torch.amp.GradScaler(self.device)

    def _init_model_dir(self):
        if not self.config['model_dir'].startswith('experiments'):
            self.config['model_dir'] = os.path.join('experiments', self.config['model_dir'])
        if not os.path.exists(self.config['model_dir']):
            print('Create model directory:', self.config['model_dir'])
            os.makedirs(self.config['model_dir'], exist_ok=True)

        with open(os.path.join(self.config['model_dir'], 'config.yml'), 'w') as f:
            yaml.dump(self.config, f)

    def _init_tensorboard(self):
        exp_name = self.config['model_dir'].split('experiments/')[1]
        log_dir = os.path.join('experiments', exp_name, 'logs')
        writer = SummaryWriter(log_dir=log_dir)
        self.writer = writer
        print('Tensorboard logs will be saved at:', log_dir)