import argparse
import numpy as np
import pandas as pd
import pickle

from omnia.generics.nas import NNIPredictor


def prepare_data(dataset_dir_list : list) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Prepare data for training, validation and testing
    :param dataset_dir_list: path to datasets to use
    :return: x_train, x_test, x_val, y_train, y_test, y_val
    """
    print('Loading data...')
    #read splits pickle file
    with open('/home/lmarreiros/drug_response_pipeline-master/almanac/data/splits/train_val_test_groups_split_inds_12321.pkl', 'rb') as handle:
        splits = pickle.load(handle)
        train_inds = splits[0]
        val_inds = splits[1]
        test_inds = splits[2]

    #read data
    y = pd.read_csv('/home/lmarreiros/drug_response_pipeline-master/almanac/data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv', sep=",", header=0, index_col=0)
    y.drop(['NAME','CELLNAME','GROUP','NSC1','NSC2','SMILES_A','SMILES_B'], axis=1, inplace=True)
    # split y into train, val, test
    y_train = y.iloc[train_inds]
    y_val = y.iloc[val_inds]
    y_test = y.iloc[test_inds]

    feature_train_df_array = []
    feature_val_df_array = []
    feature_test_df_array = []

    # read features
    for path in dataset_dir_list:
        # Split the string by backslashes and get the last part
        parts = path.split('/')
        feature_name = parts[-1]

        feature_train = np.load(path + '_train.npy')
        feature_val = np.load(path + '_val.npy')
        feature_test = np.load(path + '_test.npy')

        feature_train_df = pd.DataFrame({feature_name: [row for row in feature_train]})
        feature_val_df = pd.DataFrame({feature_name: [row for row in feature_val]})
        feature_test_df = pd.DataFrame({feature_name: [row for row in feature_test]})

        feature_train_df_array.append(feature_train_df)
        feature_val_df_array.append(feature_val_df)
        feature_test_df_array.append(feature_test_df)

    x_train_df = pd.concat(feature_train_df_array, axis=1)
    x_val_df = pd.concat(feature_val_df_array, axis=1)
    x_test_df = pd.concat(feature_test_df_array, axis=1)

    return x_train_df, x_test_df, x_val_df, y_train, y_test, y_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drug Synergy')
    parser.add_argument('--dataset_dir_list', nargs='+', type=str, help='Paths to every data to use')
    parser.add_argument('--time_budget', type=int, help='Time budget for training in seconds')
    parser.add_argument('--strategy', type=str, help='Search strategy')
    parser.add_argument('--experiment_name', type=str, help='Name of experiment')

    args = parser.parse_args()
    dataset_dir_list = args.dataset_dir_list
    time_budget = args.time_budget
    strategy = args.strategy
    experiment_name = args.experiment_name

    x_train, x_test, x_val, y_train, y_test, y_val = prepare_data(dataset_dir_list)
    print(x_train.shape)
    print(x_train.head)
    multi_input_list = []
    for path in dataset_dir_list:
        # Split the string by backslashes and get the last part
        parts = path.split('/')
        feature_name = parts[-1]
        multi_input_list.append([feature_name])

    print(multi_input_list)
    nni_predictor = NNIPredictor(x=x_train, y=y_train, x_val=x_val, y_val=y_val, time_limit=time_budget, max_epochs=500,
                                 batch_size=64, search_strategy=strategy, experiment_name=experiment_name,
                                 multi_input_list=multi_input_list)
    nni_predictor.fit(refit_best=True)
    scores = nni_predictor.score(x_test, y_test)
    print(scores)

    with open(experiment_name + '.txt', 'w') as f:
        f.write(str(scores))
