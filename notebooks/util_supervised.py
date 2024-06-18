import zipfile
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Open the zip file
def load_feature_dataset_pca(filepath='../data/df_256pc_pickle.zip'):
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        # Open the pickle file within the zip
        with zip_ref.open('df_256pc') as file:
            # Load the data using pickle
            df = pickle.load(file)
    return df

def sup_train_test_split(df):
    #get rid of unknown moa
    df = df[df['Metadata_MoA'] != 'unknown']
    
    #get stratified random sample of compounds (by MoA) as test group
    compound_moas = list(df.groupby(['Metadata_MoA', 'Metadata_Compound']).count().index)
    compound_moas = pd.DataFrame(compound_moas, columns=['MoA', 'Compound'])
    test_compounds = compound_moas.groupby('MoA', group_keys=False).apply(lambda x: x.sample(frac=.25, random_state=224))['Compound']
    compound_moas['in_testset'] = compound_moas['Compound'].isin(test_compounds)
    
    df_test = df[df['Metadata_Compound'].isin(test_compounds)]
    df_train = df[~df['Metadata_Compound'].isin(test_compounds)]
    
    train_y = df_train['Metadata_MoA']
    test_y = df_test['Metadata_MoA']
    train_X = df_train.iloc[:,:256]
    test_X = df_test.iloc[:,:256]
    return(train_X, train_y, test_X, test_y)

train_X, train_y, test_X, test_y = sup_train_test_split(load_feature_dataset_pca())

