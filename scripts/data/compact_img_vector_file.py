import argparse

import polars as pl

parser = argparse.ArgumentParser(description="Remove file and path columns from the embedding vector file")
parser.add_argument('-infile', type=str, required=True, help="Input file name, full path")
parser.add_argument('-outfile', type=str, required=True, help="Output file name, full path")
args = parser.parse_args()


def main():
    df_pl_img_embed = pl.read_csv(args.infile)

    plate_no = df_pl_img_embed.select(
        pl.col("Metadata_Plate_DAPI").str.extract(r".+_(\d+)", group_index=1).cast(pl.Int64).alias("Metadata_PlateNumber"),
    )
    
    df_pl_img_embed = pl.concat(
        [
            plate_no,
            df_pl_img_embed,
        ],
        how="horizontal",
    )

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

    out.write_csv(args.outfile)


if __name__=="__main__":
    main()