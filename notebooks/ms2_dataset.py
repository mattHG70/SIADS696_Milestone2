"""
This Python class implements the PyTorch Dataset. This Dataset is
used with a PyTroch Dataloader to load batches of images during
model trainging and evalutation. Specifically implemente for holding
the image embedding vectors.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np


class EmbedVecDataset(Dataset):
    def __init__(self, df_dataset, label, data_prefix, label_dict):
        """
        Initialization of the Dataset.
        Params:
            - df_dataset (Pandas dataframe): the dataframe containing the full dataset.
            - label (string): name of the column containing the labels, e.g. MoA.
            - data_prefix (string): the prefix of the embedding vector columns, e.g. Z.
            - label_dict (dictionary): dictionary of all labels.
         """
        self.df_data = df_dataset
        self.data_label = label
        self.data_prefix = data_prefix
        self.data_cols = [c for c in self.df_data.columns if c.startswith(self.data_prefix)]
        self.label_dict = label_dict

    def __len__(self):
        """
        Returns the length of dataset.
        """
        return self.df_data.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves an embedding vector and the label from the dataset. The label gets
        one-hot encoded.
        Params:
            - idx (int): index of the embedding vector.
        """
        moa_label = self.df_data.loc[idx, self.data_label]
        moa_tensor = torch.tensor(self.label_dict[moa_label])
        label = F.one_hot(moa_tensor, num_classes=len(self.label_dict)).type(torch.FloatTensor)

        embed_vector = self.df_data.loc[idx, self.data_cols]
        embed_tensor = torch.from_numpy(embed_vector.to_numpy().astype(np.float32))

        return embed_tensor, label