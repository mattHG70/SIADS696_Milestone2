"""
Python script used ot execute a crossvalidation gridsearch on the 
feed-forward neural network to classify MoA.
The script is run in a cluster job. A GPU can be used if available.
Due to the long runtime of the gridsearch and the job limits on Great Lakes HPC
the script is run several times with a subset of the model hyperparameters. The 
individual hyperparameters of each run are then merged for further analysis.
A complete list of hyperparameters is available in the appendix of the 
project report.
"""
import os
import argparse
import itertools
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

import ms2_model
from ms2_model import Net3072
from ms2_dataset import EmbedVecDataset


def do_loo_crossval(df, moa_dict, n_classes, parameters, random_state, device, verbose=True):
    """
    Function which executes the leave-on-compound-out crossvalidation.
    Params:
        - df (Pandas dataframe): dataframe containing the full dataset.
        - moa_dict (dictionary): dictionary of MoA.
        - n_classes (int): number of distinct MoA classes.
        - parameters (tuple): hyperparameters used in the crossvalidation.
        - random_state (int): controls randomization, ensures reproducibility.
        - device (PyTroch device): device use in the process, can be "cpu" or "gpu".
        - verbose (bool): output additionla information if set to True.
    """
    # Set hyperparameters for crossvalidation
    learning_rate = parameters[0]
    batch_size = parameters[1]
    n_epochs = parameters[2]

    # Create array to hold the crossvalidation results
    cv_splits = len(df["Metadata_Compound"].unique())
    cv_values = np.zeros((cv_splits, 4))
    cv_result = list()
    
    # Loop through the training dataset and get training and validation 
    # sets for crossvalidation
    for i, (train_set, valid_set) in enumerate(iter_loo_compound(df)):
        train_set = train_set.reset_index()
        valid_set = valid_set.reset_index()
        compound = valid_set.loc[0, "Metadata_Compound"]
        moa = valid_set.loc[0, "Metadata_MoA"]

        # Set random state to enable reproducability
        torch.manual_seed(random_state)
        
        # Create training and validation Datasets and Dataloaders
        train_dataset = EmbedVecDataset(train_set, "Metadata_MoA", "Z", moa_dict)
        valid_dataset = EmbedVecDataset(valid_set, "Metadata_MoA", "Z", moa_dict)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
        # Create NN model and set optimizer and loss function
        model = Net3072(n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
    
        cv_fold_values = np.zeros((n_epochs, 4))
        
        # Do training and validation. Iterate through the epochs
        for epoch in range(n_epochs):
            train_loss, train_acc = ms2_model.train_model(model, optimizer, loss_func, train_dataloader, device)
            valid_loss, valid_acc = ms2_model.valid_model(model, loss_func, valid_dataloader, device)
            tl = train_loss / len(train_dataloader.dataset)
            ta = train_acc / len(train_dataloader.dataset)
            vl = valid_loss / len(valid_dataloader.dataset)
            va = valid_acc / len(valid_dataloader.dataset)
            cv_fold_values[epoch,0] = tl
            cv_fold_values[epoch,1] = ta
            cv_fold_values[epoch,2] = vl
            cv_fold_values[epoch,3] = va
        cv_values[i,0] = cv_fold_values[-1,0]
        cv_values[i,1] = cv_fold_values[-1,1]
        cv_values[i,2] = cv_fold_values[-1,2]
        cv_values[i,3] = cv_fold_values[-1,3]
    
        if verbose:
            print(f"LOO {compound}: Train loss: {tl:>5f}  Train accuracy: {ta:>5f}  Valid loss: {vl:>5f}  Valid accuracy: {va:>5f}")
    
    # Calculate the mean and standrad deviation of the training and validaion
    # loss and accuracy
    cv_mean_tl = cv_values[:,0].mean()
    cv_std_tl = cv_values[:,0].std()
    cv_mean_ta = cv_values[:,1].mean()
    cv_std_ta = cv_values[:,1].std()
    cv_mean_vl = cv_values[:,2].mean()
    cv_std_vl = cv_values[:,2].std()
    cv_mean_va = cv_values[:,3].mean()
    cv_std_va = cv_values[:,3].std()
    
    if verbose:
        print(f"Training - loss mean: {cv_mean_tl:>5f}  loss std: {cv_std_tl:>5f}")
        print(f"Training - accuracy mean: {cv_mean_ta:>5f}  accuracy std: {cv_std_ta:>5f}")
        print(f"Validation - loss mean: {cv_mean_vl:>5f}  loss std: {cv_std_vl:>5f}")
        print(f"Validation - accuracy mean: {cv_mean_va:>5f}  accuracy std: {cv_std_va:>5f}")

    return [cv_mean_tl, cv_std_tl, cv_mean_ta, cv_std_ta, cv_mean_vl, cv_std_vl, cv_mean_va, cv_std_va]


def enumerate_params(param_dict):
    """
    Creates an enumerated list of hyperparametes used in the gridsearch.
    Params:
        - param_dict (dictionary): dictiionary of hyperparameters.
    """
    params = list(param_dict.keys())
    param_list = [param_dict[param] for param in params]
    param_combi =  itertools.product(*param_list) 
    
    return params, list(param_combi)


def iter_loo_compound(df):
    """
    Iterator function to implement leave-one-compound-out crossvalidation.
    Yields a training and validation dataframe.
    Params:
        - df (Pandas dataframe): dataframe containing the full dataset.
    """
    # Get a list of all compounds in the dataset
    compounds = df["Metadata_Compound"].unique().tolist()

    for compound in compounds:
        yield df.query(f'Metadata_Compound != "{compound}"'), df.query(f'Metadata_Compound == "{compound}"')


def balance_train_data(df, random_state):
    df_grp = df.groupby(["Metadata_MoA"])["Metadata_Compound"].count().reset_index(name="count")
    mean_count = int(df_grp.drop(df_grp[df_grp["Metadata_MoA"] == "DMSO"].index)["count"].mean().round())

    df_dmso = df[df["Metadata_MoA"] == "DMSO"].sample(n=mean_count, random_state=random_state)
    df_other = df.drop(df[df["Metadata_MoA"] == "DMSO"].index)

    df_all = pd.concat([df_other, df_dmso], axis=0)

    return df_all.reset_index(drop=True)


def do_gridseachcv(param_dict, moa_dict, df, n_classes, random_state, device):
    """
    Exectues the acutal gridsearch crossvalidation.
    Params:
        - param_dict (dictionary): dictiionary of hyperparameters.
        - moa_dict (dictionary): dictionary of MoA.
        - df (Pandas dataframe): dataframe containing the full dataset.
        - n_classes (int): number of distinct MoA classes.
        - random_state (int): controls randomization, ensures reproducibility.
        - device (PyTroch device): device use in the process, can be "cpu" or "gpu". 
    """
    
    # Columns holding the training and validation results
    value_cols = ["trianing_loss mean", 
                  "training_loss std", 
                  "training_acc mean", 
                  "training_acc std", 
                  "validation_loss mean", 
                  "validation_loss std", 
                  "validaton_acc mean", 
                  "validation_acc std"]
    
    # Enumerate all hyperparameters used in the gridsearch
    parameters, combinations = enumerate_params(param_dict)

    # Define the cross validation result columns
    result_cols = parameters
    result_cols.extend(value_cols)

    # Iterate through all hyperparameter combinations and execute a
    # leave-one-compound-out crossvalidaton. Store the results in a list.
    gridsearch_records = list()
    for combination in combinations:
        print(f"learning rate: {combination[0]}  batch_size: {combination[1]}  epochs: {combination[2]}")
        grid_results = do_loo_crossval(df, moa_dict, n_classes, combination, random_state, device, verbose=False)
        combi_results = list(combination)
        combi_results.extend(grid_results)
        gridsearch_records.append(tuple(combi_results))

    # Return a Pandas dataframe containing the results of the gridsearch crossvalidation.
    return pd.DataFrame.from_records(gridsearch_records, columns=result_cols)


# Parse command line arguments passed to the script.
parser = argparse.ArgumentParser(description="Crossvalidation Script for supervised NN")
parser.add_argument('-infile', type=str, required=True, help="Input file name training data, full path")
parser.add_argument('-outfile', type=str, required=True, help="Ouput file name cross validation table, full path")
args = parser.parse_args()

def main():
    """
    Main fuction of the script defining the gridsearch process.
    """
    # set the random state to ensure reproducability
    random_state = 764
    torch.manual_seed(random_state)
    
    # Check which device is available. Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read in the full training dataset
    df_data = pd.read_parquet(args.infile)

    # Create a dictionary of MoAs present in the training dataset and
    # set n_classes to the number of distinct MoAs.
    moa_list = df_data[~df_data["Metadata_MoA"].isnull()].loc[:, "Metadata_MoA"].unique().tolist()
    moa_dict = {moa: idx for moa, idx in zip(moa_list, range(len(moa_list)))}
    n_classes = len(moa_dict.keys())
    
    # Specify the model hyperparameters used in the gridsearch
    param_dict = {"learning_rate": [0.001],
                 "batch_size": [32],
                 "epochs": [15, 20]}

    # Execute the gridsearch crossvalidation
    df = do_gridseachcv(param_dict, moa_dict, df_data, n_classes, random_state, device)

    # Write the results to a parquet file
    df.to_parquet(args.outfile, index=False)


if __name__=="__main__":
    """
    Call the main function.
    """
    main()
