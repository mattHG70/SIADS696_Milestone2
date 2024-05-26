"""
This script downloads the CSV files for the BBBC021 dataset from the 
Broad Institue.
Dataset URL: https://bbbc.broadinstitute.org/BBBC021
In addtion the file path BBBC021_v1_image.csv file is set to the actual file
path of the images because this file is the basis of transfer learning step. 
"""
import os
import re
import argparse

import pandas as pd

# specify the URLs and the file names
dataset_urls =[
    ("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv","BBBC021_v1_image.csv", True),
    ("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_compound.csv","BBBC021_v1_compound.csv", False),
    ("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv","BBBC021_v1_moa.csv", False)
]


"""
Command line parameters:
- -outdir: the directory in which the downloaded files are stored
- -imgdir: the root directory of the actual image files
"""
parser = argparse.ArgumentParser(description="Download CSV file for BBBC021_v1 dataset")
parser.add_argument('-outdir', type=str, required=True, help="Output directory where the files should be stored")
parser.add_argument('-imgdir', type=str, required=True, help="Image directory for BBBC021 images")
args = parser.parse_args()


def change_img_path(p, imgpath):
    """
    Function which sets the correct path to the image.
    """
    path = re.match(r".+\/(.+)", p)
    return os.path.join(imgpath, path.group(1))


def main():
    """
    The main fuction of the script.
    The CSV files are directly read into dataframes and manipulated if needed.
    """
    for data_file  in dataset_urls:
        # read URL directly into dataframe
        df = pd.read_csv(data_file[0])

        # change the image path in the image file to the actual path of the
        # images
        if data_file[2]:
            img_path_cols = [c for c in df.columns if c.startswith("Image_Path")]
            for img_path in img_path_cols:
                df[img_path] = df[img_path].apply(change_img_path, imgpath=args.imgdir)

        # write the dataframes to CSV files
        df.to_csv(os.path.join(args.outdir, data_file[1]), index=False)

    
if __name__=="__main__":
    main()