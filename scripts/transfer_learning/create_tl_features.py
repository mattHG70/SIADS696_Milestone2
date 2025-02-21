"""
Main Python script used to generate the image embedding vectors
using transfer learning.
"""
import argparse
import os
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms

import tl_model
import tl_transform
from tl_datasets import tl_dataset
from tl_utilities import compute_features, get_device, merge_features

# Parse command line arguments passed to the script.
parser = argparse.ArgumentParser(description='Arguments of the TL features generation')
parser.add_argument('-dev', type=str, required=False, help='Which devide to use? Options are gpu or cpu')
parser.add_argument('-image_list', type=str, required=True, help='File name for the image list (or path)')
parser.add_argument('-outdir', type=str, required=False, help='Output directory for the embedding files')
parser.add_argument('-channels', type=str, required=True, help='List of channels (as names)')
args = parser.parse_args()


def main():
    """
    Main function defining the transfer learning process. Command liine arguments are
    user for customization of the process.
    """
    # Get device - use GPU if available
    device = get_device(args.dev)

    # Initialize the tl_model
    model = tl_model.DenseNet()
    model = model.to(device)
    
    # Define model transformation
    tra = [tl_transform.resize((1024,1024)), transforms.ToTensor(), tl_transform.to_float(), tl_transform.normalize()]
    
    # Get the channels and the image list file
    channels_list = args.channels.split(",")
    path_to_csv = f"{args.image_list}.csv"
    
    # Read image list data frame
    df_images = pd.read_csv(path_to_csv)
    
    # Columns of metadata
    meta_cols = [c for c in df_images.columns if not c.startswith('Image')]
    
    # Set output directory, choose current dir if parameter is None
    if args.outdir is None:
        out_dir = '.'
    else:
        out_dir = args.outdir
    
    # Process the images of each channel
    for channel in channels_list:
        print(f"Computing TL features channel: {channel}")

        # Initialize data frame of errors epr channel
        df_errors_ch = pd.DataFrame(columns = ['PathToImage', 'Channel', 'ErrorType'])
        
        # List of images path
        imgs = df_images['Image_PathName_%s' % channel] + '/' + df_images['Image_FileName_%s' % channel] 
        
        # Add the path as a column in the data frame (for current channel) - use later to merge
        df_images['Image_Files'] = imgs    
        
        # List of images with corrupted file
        imgs_errors = [img for img in imgs if not os.path.isfile(img)]
        df_errors_ch['PathToImage'] = imgs_errors
        df_errors_ch['Channel'] = [channel for _ in range(len(imgs_errors))]
        df_errors_ch['ErrorType'] = ['FileNotFound' for _ in range(len(imgs_errors))]
        
        # Images which are files - not currupted
        imgs_files = [img for img in imgs if os.path.isfile(img)]
        

        # Create dataset - with files that exist
        dataset = tl_dataset(imgs_files, transform=transforms.Compose(tra), iterations=1)
        sampler = torch.utils.data.SequentialSampler(dataset)
 
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 sampler=sampler,
                                                 batch_size=8,
                                                 num_workers=1,
                                                 pin_memory=True)


        # Get metadata from data frame
        metadata = df_images[meta_cols]
        
        # Get channel columns
        channel_data = df_images[['Image_PathName_%s' % channel, 'Image_FileName_%s' % channel]]
    
        # Original data frame all data
        df_orig = pd.concat((metadata, channel_data), axis = 1)
        
        # Add images files columns
        df_orig['ImagesList'] = imgs    
    
        # Get the features for the whole dataset
        features = compute_features(dataloader, model, device)
        
        # Creates embeddings
        embeds = [pd.Series(features[:, i], name='Z{:03d}_{}'.format(i, channel)) for i in range(features.shape[1])]
        embeds = pd.concat(embeds, axis=1)
        
        # Data frame with embeddings
        df_embed = embeds.copy()
    
        df_embed['ImagesList'] = imgs_files
        
        # Merge and get the non valuable embeddings as NaN
        df_embed = df_orig.merge(df_embed, on='ImagesList', how='left')

        df_embed = df_embed.drop(columns = ['ImagesList'])
        
        # write embedding vectors and errors per channel
        df_embed.to_csv(os.path.join(out_dir, f"tl_features_channel_{channel}.csv"), index=False)
        df_errors_ch.to_csv(os.path.join(out_dir, f"df_errors_channel_{channel}.csv"), index = False)
    
    # merge embedding vectors and write embedding vector dataset
    df_merged_all = merge_features(channels_list, out_dir)
    
    # Compile errors file
    df_errors = pd.DataFrame(columns=['PathToImage', 'Channel', 'ErrorType'])

    # Generate error report
    for channel in channels_list:
        file_errors_ch = os.path.join(out_dir, f"df_errors_channel_{channel}.csv")
        df_errors_ch = pd.read_csv(file_errors_ch)
        os.remove(file_errors_ch)
        df_errors = pd.concat((df_errors, df_errors_ch))
    
    df_errors.to_csv(os.path.join(out_dir, 'Errors_Report.csv'), index=False)
    

if __name__=="__main__":
    """
    Call the main function.
    """
    main()
    
