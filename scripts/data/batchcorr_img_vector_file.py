"""
This script will apply batch correction using PyCombat to the embedding
vectors. The PyCombat implementation from the library InMoose is used
for this task and had been installed in the Python conda environment.
Library documentation:
https://inmoose.readthedocs.io/en/latest/
"""
import argparse

import pandas as pd
from inmoose.pycombat import pycombat_norm


"""
Command line parameters:
- -infile: full path to the CSV file containing the image embedding vectors.
- -outfile: full path to which the batch corrected image embedding vector file
            will get written.
"""
parser = argparse.ArgumentParser(description="Batch correction of the embedding vector fle using PyCombat")
parser.add_argument('-infile', type=str, required=True, help="Input file name, full path")
parser.add_argument('-outfile', type=str, required=True, help="Output file name, full path")
args = parser.parse_args()


def main():
    """
    Main function of the script.
    The embedding vector file gets loaded into Pandas dataframe. The actual 
    embedding vectors (data) and the plates (batches) gets extracted from the
    dataframe. pycombat_norm is used to do the batch correction. The corrected
    vectors are concatenated to the metadata and the final dataframe gets 
    written to a CSV file.
    """

    # read in embedding file. Set smiles and moa column to datatype string
    # because Pandas finds mixed datatypes due to nan values.
    df_embed_vec = pd.read_csv(args.infile, dtype={"Metadata_SMILES": "str", "Metadata_MoA": "str"})

    # split in data and batches (plates)
    data = df_embed_vec.iloc[:,9:]
    batches = df_embed_vec.iloc[:,0]
    
    # actual batch correction step
    data_corrected = pycombat_norm(data.T, batches)

    # concatenate corrected vectors with metadata
    df_embed_vec_corr = pd.concat([df_embed_vec.iloc[:,:9], data_corrected.T], axis=1)

    # write out to CSV file
    df_embed_vec_corr.to_csv(args.outfile, index=False)
 
    
if __name__=="__main__":
    main()