import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet(nn.Module):
    """
    Python class defining the model used for the transfer learning task.
    Pre-trained model used: DenseNet121 
    """
    def __init__(self, fixed_extractor=True):
        super(DenseNet, self).__init__()
        
        # Load pretrained original densenet
        original_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # Freeze weights
        if fixed_extractor:
            for param in original_model.parameters():
                param.requires_grad = False
        
        # Adapt backend
        new_model = list(original_model.children())[:-1]
        new_model.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.features = nn.Sequential(*new_model)

        self.classifier = None
        self.top_layer = None
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
