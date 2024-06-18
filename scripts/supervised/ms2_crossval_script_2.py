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

    learning_rate = parameters[0]
    batch_size = parameters[1]
    n_epochs = parameters[2]

    cv_splits = len(df["Metadata_Compound"].unique())
    cv_values = np.zeros((cv_splits, 4))
    cv_result = list()
    
    for i, (train_set, valid_set) in enumerate(iter_loo_compound(df)):
        train_set = train_set.reset_index()
        valid_set = valid_set.reset_index()
        compound = valid_set.loc[0, "Metadata_Compound"]
        moa = valid_set.loc[0, "Metadata_MoA"]
        torch.manual_seed(random_state)
        
        train_dataset = EmbedVecDataset(train_set, "Metadata_MoA", "Z", moa_dict)
        valid_dataset = EmbedVecDataset(valid_set, "Metadata_MoA", "Z", moa_dict)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
        model = Net3072(n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
    
        cv_fold_values = np.zeros((n_epochs, 4))
        
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


# def do_kfold_crossval(n_folds, dataset, n_classes, parameters, random_state, device, verbose=True):
#     kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
#     cv_values = np.zeros((n_folds, 4))

#     learning_rate = parameters[0]
#     batch_size = parameters[1]
#     n_epochs = parameters[2]

#     for fold ,(train_idx, valid_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
#         torch.manual_seed(random_state)
        
#         train_sampler = SubsetRandomSampler(train_idx)
#         valid_sampler = SubsetRandomSampler(valid_idx)
#         train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
#         valid_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    
#         model = Net3072(n_classes)
#         model.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         loss_func = nn.CrossEntropyLoss()
    
#         cv_fold_values = np.zeros((n_epochs, 4))
        
#         for epoch in range(n_epochs):
#             train_loss, train_acc = ms2_model.train_model(model, optimizer, loss_func, train_dataloader, device)
#             valid_loss, valid_acc = ms2_model.valid_model(model, loss_func, valid_dataloader, device)
            
#             tl = train_loss / len(train_dataloader.sampler)
#             ta = train_acc / len(train_dataloader.sampler)
#             vl = valid_loss / len(valid_dataloader.sampler)
#             va = valid_acc / len(valid_dataloader.sampler)
            
#             cv_fold_values[epoch,0] = tl
#             cv_fold_values[epoch,1] = ta
#             cv_fold_values[epoch,2] = vl
#             cv_fold_values[epoch,3] = va
        
#         cv_values[fold,0] = cv_fold_values[-1,0]
#         cv_values[fold,1] = cv_fold_values[-1,1]
#         cv_values[fold,2] = cv_fold_values[-1,2]
#         cv_values[fold,3] = cv_fold_values[-1,3]

#         if verbose:
#             print(f"Fold {fold+1}: Train loss: {tl:>5f}  Train accuracy: {ta:>5f}  Valid loss: {vl:>5f}  Valid accuracy: {va:>5f}")
    
#     cv_mean_tl = cv_values[:,0].mean()
#     cv_std_tl = cv_values[:,0].std()
#     cv_mean_ta = cv_values[:,1].mean()
#     cv_std_ta = cv_values[:,1].std()
#     cv_mean_vl = cv_values[:,2].mean()
#     cv_std_vl = cv_values[:,2].std()
#     cv_mean_va = cv_values[:,3].mean()
#     cv_std_va = cv_values[:,3].std()

#     if verbose:
#         print(f"Training - loss mean: {cv_mean_tl:>5f}  loss std: {cv_std_tl:>5f}")
#         print(f"Training - accuracy mean: {cv_mean_ta:>5f}  accuracy std: {cv_std_ta:>5f}")
#         print(f"Validation - loss mean: {cv_mean_vl:>5f}  loss std: {cv_std_vl:>5f}")
#         print(f"Validation - accuracy mean: {cv_mean_va:>5f}  accuracy std: {cv_std_va:>5f}")

#     return [cv_mean_tl, cv_std_tl, cv_mean_ta, cv_std_ta, cv_mean_vl, cv_std_vl, cv_mean_va, cv_std_va]


def enumerate_params(param_dict):
    params = list(param_dict.keys())
    param_list = [param_dict[param] for param in params]
    param_combi =  itertools.product(*param_list) 
    
    return params, list(param_combi)


def iter_loo_compound(df):
    compounds = df["Metadata_Compound"].unique().tolist()
    for compound in compounds:
        # print(compound)
        yield df.query(f'Metadata_Compound != "{compound}"'), df.query(f'Metadata_Compound == "{compound}"')


def balance_train_data(df, random_state):
    df_grp = df.groupby(["Metadata_MoA"])["Metadata_Compound"].count().reset_index(name="count")
    mean_count = int(df_grp.drop(df_grp[df_grp["Metadata_MoA"] == "DMSO"].index)["count"].mean().round())

    df_dmso = df[df["Metadata_MoA"] == "DMSO"].sample(n=mean_count, random_state=random_state)
    df_other = df.drop(df[df["Metadata_MoA"] == "DMSO"].index)

    df_all = pd.concat([df_other, df_dmso], axis=0)

    return df_all.reset_index(drop=True)


def do_gridseachcv(param_dict, moa_dict, df, n_classes, random_state, device):
    value_cols = ["trianing_loss mean", 
                  "training_loss std", 
                  "training_acc mean", 
                  "training_acc std", 
                  "validation_loss mean", 
                  "validation_loss std", 
                  "validaton_acc mean", 
                  "validation_acc std"]
    parameters, combinations = enumerate_params(param_dict)

    result_cols = parameters
    result_cols.extend(value_cols)

    gridsearch_records = list()
    for combination in combinations:
        print(f"learning rate: {combination[0]}  batch_size: {combination[1]}  epochs: {combination[2]}")
        grid_results = do_loo_crossval(df, moa_dict, n_classes, combination, random_state, device, verbose=False)
        combi_results = list(combination)
        combi_results.extend(grid_results)
        gridsearch_records.append(tuple(combi_results))

    return pd.DataFrame.from_records(gridsearch_records, columns=result_cols)


parser = argparse.ArgumentParser(description="Crossvalidation Script for supervised NN")
parser.add_argument('-infile', type=str, required=True, help="Input file name training data, full path")
parser.add_argument('-outfile', type=str, required=True, help="Ouput file name cross validation table, full path")
args = parser.parse_args()

def main():
    random_state = 764
    
    torch.manual_seed(random_state)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_data = pd.read_parquet(args.infile)

    moa_list = df_data[~df_data["Metadata_MoA"].isnull()].loc[:, "Metadata_MoA"].unique().tolist()
    moa_dict = {moa: idx for moa, idx in zip(moa_list, range(len(moa_list)))}
    n_classes = len(moa_dict.keys())

    # train_dataset = EmbedVecDataset(df_data, "Metadata_MoA", "Z", moa_dict)
    # cv_splits = 5
    
    param_dict = {"learning_rate": [0.001],
                 "batch_size": [32],
                 "epochs": [15, 20]}

    
    df = do_gridseachcv(param_dict, moa_dict, df_data, n_classes, random_state, device)
    df.to_parquet(args.outfile, index=False)


if __name__=="__main__":
    main()
