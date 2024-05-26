"""
This script removes columns which are not needed for data analysis and model
building. These columns contain mainly the image file names and path names
such as Image_PathName_DAPI or Image_FileName_DAPI.
In addition an additional column containing the plate number gets added. This
plate number is used as a batch number in the PyCombat based batch correction.
The whole experiment got done over 10 weeks on 55 plates.
"""
import argparse

import polars as pl

"""
Command line paramters:
- -infile: the full path of the input file created by the transfer learnings
- -outfile: the full path of the more compact output file
"""
parser = argparse.ArgumentParser(description="Remove file and path columns from the embedding vector file")
parser.add_argument('-infile', type=str, required=True, help="Input file name, full path")
parser.add_argument('-outfile', type=str, required=True, help="Output file name, full path")
args = parser.parse_args()


def main():
    """
    The main function of the script. 
    Because of the size of the files and fact that only data manipulation
    steps are exectuted the Polars instead of Pandas was chosen. Polars has a
    much better performance compared to Pandas when it comes to large files.
    """

    # read in image vector file from transfer learning step
    df_pl_img_embed = pl.read_csv(args.infile)

    # extract the plate number
    plate_no = df_pl_img_embed.select(
        pl.col("Metadata_Plate_DAPI").str.extract(r".+_(\d+)", group_index=1).cast(pl.Int64).alias("Metadata_PlateNumber"),
    )

    # add the plate number column to the dataframe
    df_pl_img_embed = pl.concat(
        [
            plate_no,
            df_pl_img_embed,
        ],
        how="horizontal",
    )

    # create output dataframe containg the columns needed for analysis and
    # model building
    out = df_pl_img_embed.select(
        pl.col("*").exclude("Image_FileName_DAPI",
                            "Image_PathName_DAPI",
                            "Image_FileName_Tubulin",
                            "Image_PathName_Tubulin",
                            "Image_FileName_Actin",
                            "Image_PathName_Actin",
                            "Metadata_Plate_DAPI",
                            "Metadata_Well_DAPI",
                            )   
    )

    # export and write the dataframe to a CSV file
    out.write_csv(args.outfile)


if __name__=="__main__":
    main()