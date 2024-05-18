import os
import re
import argparse

import pandas as pd

parser = argparse.ArgumentParser(description="Download CSV file for BBBC021_v1 dataset")
parser.add_argument('-outfile', type=str, required=True, help="Output merged datasets to file")
parser.add_argument('-datadir', type=str, required=True, help="Data directory for BBBC021 csv files")
args = parser.parse_args()

data_files = {
    "image": "BBBC021_v1_image.csv",
    "compound": "BBBC021_v1_compound.csv",
    "moa": "BBBC021_v1_moa.csv"
}


def main():
    # load all csv files into dataframes
    df_images = pd.read_csv(os.path.join(args.datadir, data_files["image"]))
    df_compound = pd.read_csv(os.path.join(args.datadir, data_files["compound"]))
    df_moa = pd.read_csv(os.path.join(args.datadir, data_files["moa"]))

    # merge images and compounds
    df_bbbc021 = df_images.merge(df_compound, left_on="Image_Metadata_Compound", right_on="compound")
    df_bbbc021 = df_bbbc021.drop(columns=["compound"])

    # prepare moa dataset: drop concentration and duplicates
    df_moa = df_moa.drop(columns=["concentration"])
    df_moa = df_moa.drop_duplicates()

    # merge into final dataset
    df_bbbc021 = df_bbbc021.merge(df_moa, left_on="Image_Metadata_Compound", right_on="compound", how="left")
    df_bbbc021 = df_bbbc021.drop(columns=["compound"])

    # write final dataset
    df_bbbc021.to_csv(os.path.join(args.datadir, args.outfile), index=False)
    
if __name__=="__main__":
    main()