import numpy as np
import pandas as pd 
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

class VIST(Dataset):
    """ VIST Dataset """

    def __init__ (self, annotations_file, root_dir="/images/train", transform=None):
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
    def __init__():
        pass

    def sample():
        pass

