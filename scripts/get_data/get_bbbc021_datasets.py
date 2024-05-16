import os
import re
import requests
import argparse

import pandas as pd

dataset_urls =[
    ("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv","BBBC021_v1_image.csv", True),
    ("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_compound.csv","BBBC021_v1_compound.csv", False),
    ("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv","BBBC021_v1_moa.csv", False)
]

parser = argparse.ArgumentParser(description="Download CSV file for BBBC021_v1 dataset")
parser.add_argument('-outdir', type=str, required=True, help="Output directory where the files should be stored")
parser.add_argument('-imgdir', type=str, required=True, help="Image directory for BBBC021 images")
args = parser.parse_args()

def change_img_path(p, imgpath):
    path = re.match(r".+\/(.+)", p)
    return os.path.join(imgpath, path.group(1))

def main():
    
    for data_file  in dataset_urls:
        df = pd.read_csv(data_file[0])

        if data_file[2]:
            img_path_cols = [c for c in df.columns if c.startswith("Image_Path")]
            for img_path in img_path_cols:
                df[img_path] = df[img_path].apply(change_img_path, imgpath=args.imgdir)
        
        df.to_csv(os.path.join(args.outdir, data_file[1]), index=False)
    
    print("files written")
    
if __name__=="__main__":
    main()