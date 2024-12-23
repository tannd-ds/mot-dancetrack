import time
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class TrackingDataset(Dataset):
    def __init__(self, path, config=None):
        self.config = config.copy()
        self.path = path
        self.set = path.split("/")[-1]

        tic = time.time()
        self.aug_random_var = 0.001
        self.config["augment_data"] = self.config.get("augment_data", False)
        # Force disable data augmentation for val set
        if self.set == 'val':
            print("Disabling data augmentation for validation set.")
            self.config["augment_data"] = False

        # Default interval is 5
        self.interval = self.config.get("interval", 5)

        self.trackers = {}
        self.data = []  

        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a valid directory.")

        self.seqs = [s for s in os.listdir(path) if not s.startswith('.') and "gt_t" not in s]
        for seq in self.seqs:
            trackerPath = os.path.join(path, seq, "img1/*.txt")
            self.trackers[seq] = sorted(glob.glob(trackerPath))

            for pa in self.trackers[seq]:
                gt = np.loadtxt(pa, dtype=np.float32)
                self.precompute_data(seq, gt)  # Precompute data for this sequence

        print(f"Loaded {len(self.data)} items in {time.time() - tic:.2f}s")

    def precompute_data(self, seq, track_gt):
        """
        Precompute and store data for the dataset.

        Parameters
        ----------
            seq: str
                The sequence name.
            track_gt: ndarray
                Ground truth data for the sequence.
        """
        boxes = track_gt[:, 2:6]
        deltas = np.diff(boxes, axis=0)
        conds = np.concatenate([boxes[:-1], deltas], axis=1)

        for init_index in range(len(track_gt) - self.interval - 1):
            curr_idx = init_index + self.interval

            data_item = {
                "cur_gt": track_gt[curr_idx],  # ndarray (9, )
                "cur_bbox": track_gt[curr_idx, 2:6],  # ndarray (4, )
                "condition": conds[init_index:curr_idx],  # ndarray (4, 8)
                "delta_bbox": deltas[curr_idx, :],  # ndarray (4, )
                "width": track_gt[curr_idx, 7],  # float
                "height": track_gt[curr_idx, 8],  # float
            }

            self.data.append(data_item)

    def augment_data(self, boxes):
        """Augment the data item, by offset the boxes by a small random value."""
        boxes = np.array(boxes)
        xywh = boxes[:, :4]
        xywh += np.random.normal(0, self.aug_random_var, xywh.shape)

        boxes[1:, :4] = xywh[1:]
        delta_xywh = boxes[:, 4:]
        delta_xywh[1:, :] = xywh[1:] - xywh[:-1]
        boxes[:, 4:] = delta_xywh
        return boxes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # if self.config["augment_data"]:
        #     data = self.data[index].copy()
        #     data['condition'] = self.augment_data(data['condition'])
        #     return data
        return self.data[index]

    def show_image(self, index):
        """Display the image at the given index using PIL."""
        # Get the image path from the dataset
        image_path = self.data[index]['image_path']
        
        # Open the image using PIL
        img = Image.open(image_path)

            
        # Display the image with matplotlib
        plt.imshow(img)
        plt.axis("off")  # Hide axis for a cleaner look
        plt.show()
        
        # Display the image
        img.show()

def augment_data(boxes, aug_random_var=0.001):
    noise = torch.randn_like(boxes[:, :, :4]) * aug_random_var
    boxes[:, :, :4] += noise.to(boxes.device)
    boxes[:, 1:, 4:] = boxes[:, 1:, :4] - boxes[:, :-1, :4]
    return boxes


def custom_collate_fn(batch):
    for sample in batch:
        if 'image_path' in sample:
            del sample['image_path']
    return torch.utils.data.default_collate(batch)


# if __name__ == '__main__':
#     dataset = DiffMOTDataset("/home/tanndds/my/datasets/dancetrack/trackers_gt_t/train")
#     print(len(dataset))
#     print(dataset[0])
#     dataset.show_image(0)