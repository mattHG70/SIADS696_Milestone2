"""
Python module containing fucntions used in the unsupervised part
of our project.
"""
import pandas as pd
from sklearn.decomposition import PCA

from inmoose.pycombat import pycombat_norm

# mapping colors to MoAs. Used in UMAP and t-SNE plots.
color_map = {
    "Unknown": "lightgray",
    "Protein degradation": "firebrick",
    "Aurora kinase inhibitors": "mediumblue",
    "Eg5 inhibitors": "aqua",
    "Epithelial": "sandybrown",
    "Kinase inhibitors": "lawngreen",
    "Protein synthesis": "teal",
    "DNA replication": "darkviolet",
    "DNA damage": "lightseagreen",
    "Microtubule destabilizers":"yellow",
    "Actin disruptors": "magenta",
    "Microtubule stabilizers": "crimson",
    "Cholesterol-lowering": "darkgreen"
}
    

def apply_combat(df, batch_col="Metadata_Plate_DAPI", data_col_pref="Z"):
    """
    Apply batch correction to the embedding vector dataset. Using the pycombat_norm
    function from the inmoose library.
    Params:
        - df (Pandas dataframe): dataframe containing metadata and the embedding vectors.
        - batch_col (string): column name of the batch column, e.g. plates.
        - data_col_prefix (string): prefix of the embedding vector columns, e.g. Z.
    """
    batches = df.loc[:, batch_col]
    data_cols = [c for c in df.columns if c.startswith(data_col_pref)]
    meta_cols = [c for c in df.columns if c not in data_cols]
    data = df[data_cols]
    metadata = df[meta_cols]

    # Apply batch correction using pycmbat
    data_comb = pycombat_norm(data.T, batches)

    # Concat the metadata and batch corrected data columns
    df = pd.concat([metadata, data_comb.T], axis=1)
    
    return df


def apply_pcawhite(df, n_components=256, data_col_pref="Z"):
    """
    Apply PCA with whitening to the embedding vectors.
    Params:
        - df (Pandas dataframe): dataframe containing metadata and the embedding vectors.
        - n_components (int): number of principal components, default 256.
        - data_col_prefix (string): prefix of the embedding vector columns, e.g. Z.
    """
    # Define the PCA columns 
    pca_cols = [f"PC{n:04}" for n in range(1, n_components+1)]
    data_cols = [c for c in df.columns if c.startswith(data_col_pref)]
    meta_cols = [c for c in df.columns if c not in data_cols]

    data = df[data_cols]
    metadata = df[meta_cols]

    # Get the DMSO embedding vectors
    data_dmso = df[df["Metadata_Compound"] == "DMSO"][data_cols]
    
    # Apply PCA with whitening
    pca = PCA(n_components=n_components, whiten=True)

    # Fit the PCA using the DMSO embedding vectors
    pca.fit(data_dmso)

    # Transform all embedding vectors
    data_pca = pca.transform(data)

    # Construct the dataframe containing the metadata and the transformed 
    # embedding vectors
    df = pd.concat([metadata, pd.DataFrame(data=data_pca, columns=pca_cols)], axis=1)

    return df
    

def collaps_well(df, group_cols=["Metadata_Plate_DAPI"], data_col_pref="Z", method="mean"):
    """
    Group embedding vectors by well using various grouping fucntions.
    Params:
        - df (Pandas dataframe): dataframe containing metadata and the embedding vectors.
        - group_cols (list): columns containing the plate data and well data.
        - data_col_prefix (string): prefix of the embedding vector columns, e.g. Z.
        - method (string): aggregation method, defaul mean.
    """
    embed_cols = [c for c in df.columns if c.startswith(data_col_pref)]
    
    df_well_level = None
    if method == "mean":
        df_well_level = df.groupby(group_cols)[embed_cols].mean().reset_index()
    elif method == "median":
        df_well_level = df.groupby(group_cols)[embed_cols].median().reset_index()
    elif method == "avg":
        df_well_level = df.groupby(group_cols)[embed_cols].mean().reset_index()
    else:
        df_well_level = df.groupby(group_cols)[embed_cols].mean().reset_index()

    return df_well_level


def load_embed_file(file, img_cols=False):
    """
    Load the file containing the embedding vectors into a Pandas dataframe.
    Params:
        - file (string): file (full path) containing the vectors.
        - img_cols (bool): flag whether certain columns should be removed.
    """
    # Read CSV file and explicity set the data types for the SMILES and MoA column
    df = pd.read_csv(file, dtype={"Metadata_SMILES": "str", "Metadata_MoA": "str"})
    
    # Remove columms prefixed with Image_
    if img_cols:
        img_cols = [c for c in df.columns if c.startswith("Image_")]
        df = df.drop(columns=img_cols)
    
    return df


def add_moa(df, moa_df=None, moa_col="Metadata_MoA", cmpd_col="Metadata_Compound"):
    """
    Add MoA column to dataframe. Sets MoA to Unknonw in case it's None.
    Params:
        - df (Pandas dataframe): dataframe containing metadata and the embedding vectors.
        - moa_df (Pandas dataframe): dataframe containing MoA labels.
        - moa_col (string): column name of the MoA labels.
        - cmpd_col (string): compound column used to merge the dataframes.
    """
    moa_df[moa_col] = moa_df[moa_col].apply(lambda m: "Unknown" if m is None else m)
    df = moa_df.merge(df, on=cmpd_col)

    return df