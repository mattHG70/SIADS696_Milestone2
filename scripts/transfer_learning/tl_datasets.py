"""
This Python class implements the PyTorch Dataset. This Dataset is
used with a PyTroch Dataloader to load batches of images during
model trainging and evalutation.
"""
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class tl_dataset(Dataset):
    def __init__(self, imgs, transform=None, iterations=1):
        
        """
        Initialization of the tl_dataset class.
        Params:
            imgs (list): list of image paths.
            transform (callable): optional transformations to be applied on a sample.
        """
        
        super(tl_dataset, self).__init__()
        
        # set transformations
        self.transform = transform
        
        # Create list of images (path)
        self.imgs = imgs

            
    def __getitem__(self, index):
        """
        This function is called by the Dataloader to get images (items) from
        the dataset. The greyscale image gets transformed into a RGB image to match
        the type of images the original model was trained on. Transformations are 
        then applied to the image.
        Params:
            index (int) = index of the image in the list.
        """
        img_name = self.imgs[index]

        img = np.asarray(Image.open(img_name))
        image = np.zeros((img.shape[0], img.shape[1], 3), dtype="float")
        image[:, :, 0] = img.copy()
        image[:, :, 1] = img.copy()
        image[:, :, 2] = img.copy()
        image = self.transform(image)
        
        return image

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.imgs)
