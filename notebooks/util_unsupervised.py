import pandas as pd
from sklearn.decomposition import PCA

from inmoose.pycombat import pycombat_norm

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


def batch_correction(df, combat=True, pcawhite=True):
    pass
    

def apply_combat(df, batch_col="Metadata_Plate_DAPI", data_col_pref="Z"):
    batches = df.loc[:, batch_col]
    data_cols = [c for c in df.columns if c.startswith(data_col_pref)]
    meta_cols = [c for c in df.columns if c not in data_cols]
    data = df[data_cols]
    metadata = df[meta_cols]

    data_comb = pycombat_norm(data.T, batches)

    df = pd.concat([metadata, data_comb.T], axis=1)
    
    return df


def apply_pcawhite(df, n_components=64, data_col_pref="Z"):
    pca_cols = [f"PC{n:04}" for n in range(1, n_components+1)]
    data_cols = [c for c in df.columns if c.startswith(data_col_pref)]
    meta_cols = [c for c in df.columns if c not in data_cols]

    data = df[data_cols]
    metadata = df[meta_cols]
    data_dmso = df[df["Metadata_Compound"] == "DMSO"][data_cols]
    
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(data_dmso)
    data_pca = pca.transform(data)

    df = pd.concat([metadata, pd.DataFrame(data=data_pca, columns=pca_cols)], axis=1)

    return df
    

def collaps_well(df, group_cols=["Metadata_Plate_DAPI"], data_col_pref="Z", method="mean"):
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
    df = pd.read_csv(file, dtype={"Metadata_SMILES": "str", "Metadata_MoA": "str"})
    if img_cols:
        img_cols = [c for c in df.columns if c.startswith("Image_")]
        df = df.drop(columns=img_cols)
    
    return df


def add_moa(df, moa_df=None, moa_col="Metadata_MoA", cmpd_col="Metadata_Compound"):
    moa_df[moa_col] = moa_df[moa_col].apply(lambda m: "Unknown" if m is None else m)
    df = moa_df.merge(df, on=cmpd_col)

    return df