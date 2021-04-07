import numpy as np
import pandas as pd 
import os 
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

class VIST(Dataset):
    """ VIST Dataset """

    def __init__ (self, annotations_file, root_dir="images/train", transform=None):
        self.labels = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__ (self):
        return len(self.labels)

    def __getitem__ (self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx,0])
        image = io.imread(img_name)
        label = self.labels.iloc[idx, 1]
        sample = {"image": image, "label": label}
        
        return sample


class Expert():
    def __init__(self):
        self.root_dir = "images/train"
        img_1 = io.imread(os.path.join(self.root_dir, "4220549.jpg"))
        img_2 = io.imread(os.path.join(self.root_dir, "4220550.jpg"))
        img_3 = io.imread(os.path.join(self.root_dir, "4220551.jpg"))
        img_4 = io.imread(os.path.join(self.root_dir, "4220553.jpg"))
        img_5 = io.imread(os.path.join(self.root_dir, "4220556.jpg"))
        img_6 = io.imread(os.path.join(self.root_dir, "4220557.jpg"))
        img_1 = transform.resize(img_1, (240,240,3), anti_aliasing=True)
        img_2 = transform.resize(img_2, (240,240,3), anti_aliasing=True)
        img_3 = transform.resize(img_3, (240,240,3), anti_aliasing=True)
        img_4 = transform.resize(img_4, (240,240,3), anti_aliasing=True)
        img_5 = transform.resize(img_5, (240,240,3), anti_aliasing=True)
        img_6 = transform.resize(img_6, (240,240,3), anti_aliasing=True)
        self.images = [img_1,img_2,img_3,img_4,img_5,img_6]

    def sample(self, batch_size=16):
        # convert to numpy array
        return np.transpose(np.stack(self.images), (0,3,1,2))

