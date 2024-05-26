"""
This script merges all 3 CSV files of the BBBC021 dataset into a final dataset
to be used in the Milestone 2 project. After the merge column names are changed
to separate between metadata and image files. In addition the columns are
reorderd: first metadata than image files. This will enable direct input to
the transfer learning step.
"""
import os
import re
import argparse

import pandas as pd

"""
Command line parameters:
- -outfile: full path of the final CSV file.
- -datadir: directoy in which the 3 separate downloaded CSV file reside.
"""
parser = argparse.ArgumentParser(description="Merge the CSV files of dataset BBBC021")
parser.add_argument('-outfile', type=str, required=True, help="Output merged datasets to file")
parser.add_argument('-datadir', type=str, required=True, help="Data directory for BBBC021 csv files")
args = parser.parse_args()

# dictionary holding the filenames
data_files = {
    "image": "BBBC021_v1_image.csv",
    "compound": "BBBC021_v1_compound.csv",
    "moa": "BBBC021_v1_moa.csv"
}


def main():
    """
    Main function of the script.
    Takes all 3 CSV files and merges them to one final CSV file. Changes
    column names and reorder columns to fit to the transfer learning script.
    """
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

    # format final dataset
    # rename metadata columns
    df_bbbc021 = df_bbbc021.rename(columns={"smiles": "Metadata_SMILES", 
                                        "moa": "Metadata_MoA", 
                                        "Replicate": "Metadata_Replicate", 
                                        "TableNumber": "Metadata_TableNumber", 
                                        "ImageNumber": "Metadata_ImageNumber", 
                                        "Image_Metadata_Plate_DAPI": "Metadata_Plate_DAPI", 
                                        "Image_Metadata_Well_DAPI": "Metadata_Well_DAPI", 
                                        "Image_Metadata_Compound": "Metadata_Compound", 
                                        "Image_Metadata_Concentration": "Metadata_Concentration"})

    # get different column chunks and order them, so that the image columns
    # such as Image_FileName and Image_PathName are moved to the end and all other 
    # data and metadata is moved to the begining
    metadata_cols = [mc for mc in df_bbbc021.columns if "Metadata" in mc]
    image_cols = [ic for ic in df_bbbc021.columns if "Image_" in ic]
    final_cols = ["Metadata_ID"]
    final_cols.extend(metadata_cols)
    final_cols.extend(image_cols)

    # add an Image_ID composed of Table- and ImageNumber
    df_bbbc021["Metadata_ID"] = df_bbbc021.apply(lambda r: str(r["Metadata_TableNumber"])+"_"+str(r["Metadata_ImageNumber"]), axis=1)

    # reorder the columns
    df_bbbc021 = df_bbbc021[final_cols]
    
    # write final dataset
    df_bbbc021.to_csv(os.path.join(args.datadir, args.outfile), index=False)


if __name__=="__main__":
    main()