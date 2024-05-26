import argparse

import pandas as pd
from inmoose.pycombat import pycombat_norm


parser = argparse.ArgumentParser(description="Batch correction of the embedding vector fle using PyCombat")
parser.add_argument('-infile', type=str, required=True, help="Input file name, full path")
parser.add_argument('-outfile', type=str, required=True, help="Output file name, full path")
args = parser.parse_args()


def main():
    df_embed_vec = pd.read_csv(os.path.join(data_dir, "bbbc021_image_embed_compact.csv"), dtype={"Metadata_SMILES": "str", "Metadata_MoA": "str"})

    data = df_embed_vec.iloc[:,9:]
    batches = df_embed_vec.iloc[:,0]
    
    data_corrected = pycombat_norm(data.T, batches)

    df_embed_vec_corr = pd.concat([df_embed_vec.iloc[:,:9], data_corrected.T], axis=1)

    df_embed_vec_corr.to_csv(os.path.join(data_dir, "bbbc021_image_embed_compact_batchcorr.csv"), index=False)
    
if __name__=="__main__":
    main()