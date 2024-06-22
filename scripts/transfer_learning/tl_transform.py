"""
This Python module defines a couple of classes which are used in the 
image embedding generation process. These classes are used to transform
the input cell images.
"""
import torch
import skimage.transform
import numbers


class to_float(object):
    """
    Transforms the integer pixel values into float values. Because the 
    transfer learning process is implemented in PyTorch the pixel values are
    directly transformed into float tensor.
    """
    def __call__(self, img):
        """
        Params:
            - img (image) = image the transformation is applied to.
        """
        img = img.type(torch.FloatTensor)
        return img


class resize(object):
    """
    Resize the images to a defined size. In our project this class is used to 
    resize the images to square 1024 x 1024 images leading to vectors of 
    size 1024.
    """
    def __init__(self, size):
        """
        Params:
            - size (int) = the size the image gets transformed into.
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), 3)
        else:
            self.size = size

    def __call__(self, img):
        """
        Params:
            - img (image) = image the transformation is applied to.
        """
        img = skimage.transform.resize(img, self.size, mode='reflect', anti_aliasing=True)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class scale(object):
    """
    Scale images to float values between 0  and 1.
    """
    def __call__(self, img):
        
        img = img/img.max()
        return img    
    
    
class normalize(object):
    """
    Normalize the images to have 0 mean and standard deviation of 1.
    """
    def __call__(self, img):
        
        mean = img.mean()
        std = img.std()
        img = (img-mean)/std
        return img