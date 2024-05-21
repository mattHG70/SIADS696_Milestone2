import os
import torch
import numpy as np
import pandas as pd


def get_device(device = None):
    
    """
    Define device to use - if available uses the GPU
    
    Params:
        - device: str, device to use. Available options: "cpu" or "cuda". Default = None, then GPU is used if available
    
    """
    
    # If the device is not defined by the user, the default option is to use GPU if available, else CPU
    if not device:
        # Do we have a GPU available?
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    print('We are using the device: {}'.format(torch.cuda.device_count()))
    print('GPU Device: ', torch.cuda.current_device())
    print('GPU device name: ', torch.cuda.get_device_name(0))
    
    return device


def compute_features(dataloader, model, device):

    model.eval()
    
    # discard the label information in the dataloader
    feature_list = []

    with torch.no_grad():
        for i, input_tensor in enumerate(dataloader):
            input_tensor = input_tensor.to(device)
            aux = model(input_tensor).data.cpu().numpy()
            feature_list.append(aux)
            if (i % 100) == 0:
                print('{0} / {1}'.format(i, len(dataloader)))
               
    features = np.concatenate(feature_list, axis=0)
    return features


def merge_features(channels_list, out_dir):
    
    # Merged data frame - just concatenate by channel - list of dataframes per channel
    df_merged = []
    
    # Load the data frame of fetaures per each channel
    for channel in channels_list:
    
        df_cur = pd.read_csv('%s/tl_features_channel_%s.csv' % (out_dir, channel))
        df_merged.append(df_cur)
    
    # Get the metadata part - same for all channels
    df_meta_merged = df_merged[0].iloc[:,0:11]
    for i in range(len(channels_list)):
        # Metadata per channel
        df_meta_ch = df_merged[i].iloc[:,11:13]
        # Concatenate the metadata per channel
        df_meta_merged = pd.concat((df_meta_merged, df_meta_ch), axis = 1)
    
    # Get the features per channel
    df_feat_merged = []
    for i in range(len(channels_list)):
        # Features per channel
        df_feat_ch = df_merged[i].iloc[:,13:]
        df_feat_merged.append(df_feat_ch)
    
    # Concatenate the features across channels
    df_feat_merged = pd.concat(df_feat_merged, axis = 1)
    
    # Get the final data frame - features + channel
    df_merged_all = pd.concat((df_meta_merged, df_feat_merged), axis = 1)
    
    df_merged_all.to_csv('%s/tl_features.csv' % out_dir, index = False)
    
    for channel in channels_list:
        
        os.remove('%s/tl_features_channel_%s.csv' % (out_dir, channel))
    
    return df_merged_all
