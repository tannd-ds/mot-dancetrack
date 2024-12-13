from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import DiffMOTDataset, custom_collate_fn
from models.simple import SimpleDiffMOTModel, TransformerDiffMOTModelV2, ResidualDiffMOTModel
from utils import calculate_iou, calculate_ade, original_shape

batch_size = 512
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/Dec13_19-56-22_LAPTOP-1RBG8HEE')

# Initialize the dataset and data loader
train_path = "/home/tanndds/my/datasets/dancetrack/trackers_gt_t/train"
val_path = "/home/tanndds/my/datasets/dancetrack/trackers_gt_t/val"
train_dataset = DiffMOTDataset(train_path)
val_dataset = DiffMOTDataset(val_path)

data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True,
)
val_data_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True,
)
print(f"Number of samples in the Train dataset: {len(train_dataset)}")
print(f"Number of samples in the validation dataset: {len(val_dataset)}")

# Define the model, loss function, and optimizer
from_epoch = 109
model = TransformerDiffMOTModelV2().to(device)
model.load_state_dict(torch.load(f"experiments/transformers_c/{from_epoch}.pth"))
print('Continue from epoch ', from_epoch+1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0025)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
scaler = torch.amp.GradScaler(device)

def step(data_loader, train=True):
    model.train() if train else model.eval()
    epoch_loss = 0

    total_iou = 0
    total_ade = 0

    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        conditions = batch['condition'].float()
        delta_bbox = batch['delta_bbox'].float()

        with torch.amp.autocast('cuda'):
            predicted_delta_bbox = model(conditions.to(device))
            loss = criterion(predicted_delta_bbox, delta_bbox.to(device))

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # MeanIoU and MeanADE
        prev_bbox = conditions[:, -1, :4]
        pred_bbox = prev_bbox + predicted_delta_bbox.cpu()
        target_bbox = batch['cur_bbox']
        w, h = batch['width'], batch['height']

        orig_pred = original_shape(pred_bbox, w, h)
        orig_target = original_shape(target_bbox, w, h)

        # Calculate IoU
        total_iou += calculate_iou(orig_pred, orig_target)
        total_ade += calculate_ade(orig_pred, orig_target)

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.8f}, MeanIoU: {total_iou/len(data_loader):.8f}, MeanADE: {total_ade/len(data_loader):.8f}")
    if train:
        writer.add_scalar("Loss/train", epoch_loss/len(data_loader), epoch)
        writer.add_scalar("MeanIoU/train", total_iou/len(data_loader), epoch)
        writer.add_scalar("MeanADE/train", total_ade/len(data_loader), epoch)
    else:
        writer.add_scalar("Loss/val", epoch_loss/len(data_loader), epoch)
        writer.add_scalar("MeanIoU/val", total_iou/len(data_loader), epoch)
        writer.add_scalar("MeanADE/val", total_ade/len(data_loader), epoch)

# Training loop
for curr_epoch in range(num_epochs):
    epoch = curr_epoch + from_epoch + 1
    step(data_loader, train=True)

    if (epoch+1) % 10 == 0:
        step(val_data_loader, train=False)

        # Save the model
        # torch.save(model.state_dict(), "../DiffMOT/diffmot_model.pth")
        torch.save(model.state_dict(), f"experiments/transformers_c/{epoch}.pth")
        print("Model saved!")

    scheduler.step()

writer.flush()
writer.close()