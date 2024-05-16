from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image


class tl_dataset(Dataset):
    def __init__(self, imgs, transform=None, iterations=1):
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            channel: name of channel to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        super(tl_dataset, self).__init__()
        
        self.transform = transform
        
        # Create list of images - path to images
        self.imgs = imgs

            
    def __getitem__(self, index):
        img_name = self.imgs[index]

        img = np.asarray(Image.open(img_name))
        image = np.zeros((img.shape[0], img.shape[1], 3), dtype="float")
        image[:, :, 0] = img.copy()
        image[:, :, 1] = img.copy()
        image[:, :, 2] = img.copy()
        image = self.transform(image)
        
        return image

    def __len__(self):
        return len(self.imgs)
