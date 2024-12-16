import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class DiffMOTDataset(Dataset):
    def __init__(self, path, config=None):
        self.config = config.copy()
        self.path = path

        self.config["augment_data"] = self.config.get("augment_data", False)
        if path == os.path.join(self.config['data_dir'], 'val'):
            self.config["augment_data"] = False
            print("Augment data is disabled for the validation dataset.")

        try:
            self.interval = self.config.interval + 1
        except:
            self.interval = 4 + 1 + 5

        self.trackers = {}
        self.data = []  

        if os.path.isdir(path):
            self.seqs = [s for s in os.listdir(path) if not s.startswith('.') and "gt_t" not in s]
            self.seqs.sort()
            
            for seq in self.seqs:
                trackerPath = os.path.join(path, seq, "img1/*.txt")
                self.trackers[seq] = sorted(glob.glob(trackerPath))
                
                for pa in self.trackers[seq]:
                    gt = np.loadtxt(pa, dtype=np.float32)

                    self.precompute_data(seq, gt)  # Precompute data for this sequence

    def precompute_data(self, seq, track_gt):
        """Precompute and store data for the dataset."""
        for init_index in range(len(track_gt) - self.interval):
            cur_index = init_index + self.interval
            cur_gt = track_gt[cur_index]
            cur_bbox = cur_gt[2:6]

            boxes = [track_gt[init_index + tmp_ind][2:6] for tmp_ind in range(self.interval)]
            delt_boxes = [boxes[i+1] - boxes[i] for i in range(self.interval - 1)]
            conds = np.concatenate((np.array(boxes)[1:], np.array(delt_boxes)), axis=1)

            delt = cur_bbox - boxes[-1]

            width, height = cur_gt[7:9]
            image_path = self.path.replace("/trackers_gt_t", "") + f"/{seq}/img1/{int(cur_gt[1]):08d}.jpg"


            data_item = {
                "cur_gt": cur_gt, # ndarray (9, )
                "cur_bbox": cur_bbox,  # ndarray (4, )
                "condition": conds,  # ndarray (4, 8)
                "delta_bbox": delt,  # ndarray (4, )
                "width": width,  # float
                "height": height,  # float
                "image_path": image_path,  # str
            }

            self.data.append(data_item)

    @staticmethod
    def augment_data(boxes):
        """Augment the data item, by offset the boxes by a small random value."""
        boxes = np.array(boxes)
        xywh = boxes[:, :4]
        xywh += np.random.normal(0, 0.0010, xywh.shape)
        boxes[1:, :4] = xywh[1:]
        delta_xywh = boxes[:, 4:]
        delta_xywh[1:, :] = xywh[1:] - xywh[:-1]
        boxes[:, 4:] = delta_xywh
        return boxes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.config["augment_data"]:
            data = self.data[index].copy()
            data['condition'] = self.augment_data(data['condition'])
            return data
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