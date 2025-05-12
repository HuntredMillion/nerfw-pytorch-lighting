import torch
from torch.utils.data import IterableDataset
import imageio
import numpy as np

class NotreDameDataset(IterableDataset):
    """
    Expects a TXT file where each line is:
      <img_index> <pose(16 floats)> <lighting(one-hot + scalar)>
    and images named <img_index>.png in the img_dir.
    """
    def __init__(self, txt_file, img_dir):
        super().__init__()
        self.txt_file = txt_file
        self.img_dir  = img_dir

    def __iter__(self):
        with open(self.txt_file, 'r') as f:
            for line in f:
                vals = line.strip().split()
                idx   = vals[0]
                pose  = np.array(vals[1:17], dtype=np.float32)
                light = np.array(vals[17:], dtype=np.float32)
                img   = imageio.imread(f"{self.img_dir}/{idx}.png") / 255.0
                yield {
                    'image': torch.from_numpy(img).permute(2,0,1).float(),
                    'pose' : torch.from_numpy(pose),
                    'light': torch.from_numpy(light)
                }
