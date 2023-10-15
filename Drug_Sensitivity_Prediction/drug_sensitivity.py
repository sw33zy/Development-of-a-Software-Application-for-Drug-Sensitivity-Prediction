import os
import argparse
import numpy as np
import pandas as pd

from omnia.generics.nas import NNIPredictor
from omnia.compounds.feature_extraction.fingerprints import LayeredFingerprint, AtomPairFingerprint, MACCSKeysFingerprint, MorganFingerprint, TopologicalFingerprint
from omnia.compounds.feature_extraction.mol2vec_embeddings import Mol2VecEmbeddings


def prepare_data(dataset_dir: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Prepare data for training and testing
    :param dataset_dir: path to dataset
    :return: x_train, x_test, y_train, y_test
    """
    print('Loading data...')
    dataset_name = os.path.split(dataset_dir)[-1]
    if "onehot" in dataset_dir:
        x_train = np.load(os.path.join(dataset_dir, 'train_' + dataset_name + '_X.npy'))
        y_train = np.load(os.path.join(dataset_dir, 'train_' + dataset_name + '_y.npy'))
        x_test = np.load(os.path.join(dataset_dir, 'test_' + dataset_name + '_X.npy'))
        y_test = np.load(os.path.join(dataset_dir, 'test_' + dataset_name + '_y.npy'))

    else:
        train_filepath = os.path.join(dataset_dir, 'train_' + dataset_name + '.csv')
        test_filepath = os.path.join(dataset_dir, 'test_' + dataset_name + '.csv')

        train_data = pd.read_csv(train_filepath, sep=",", header=0, index_col=0)
        train_data.drop(columns=['ids', 'X'], inplace=True)
        x_train = train_data.drop(columns=['y'])
        y_train = train_data['y']

        test_data = pd.read_csv(test_filepath, sep=",", header=0, index_col=0)
        test_data.drop(columns=['ids', 'X'], inplace=True)
        x_test = test_data.drop(columns=['y'])
        y_test = test_data['y']

    return x_train, x_test, y_train, y_test


def featurize_data(featurizer_name: str, x_train: pd.DataFrame, x_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Featurize data
    :param featurizer_name: name of featurizer
    :param x_train: training data
    :param x_test: testing data
    :return: featurized training and testing data
    """
    print('Featurizing data...')
    featurizer = None
    if featurizer_name == 'AtomPair':
        featurizer = AtomPairFingerprint(n_bits=1024)
    elif featurizer_name == 'LayeredFP':
        featurizer = LayeredFingerprint(size=1024)
    elif featurizer_name == 'MACCS':
        featurizer = MACCSKeysFingerprint()
    elif featurizer_name == 'ECFP4':
        featurizer = MorganFingerprint(radius=2, size=1024)
    elif featurizer_name == 'ECFP6':
        featurizer = MorganFingerprint(radius=3, size=1024)
    elif featurizer_name == 'RDKitFP':
        featurizer = TopologicalFingerprint(size=1024)
    elif featurizer_name == 'Mol2vec':
        featurizer = Mol2VecEmbeddings()

    featurized_train, _ = featurizer.fit_transform(x_train)
    featurized_test, _ = featurizer.transform(x_test)

    return featurized_train, featurized_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Drug Example')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--featurizer', type=str, help='Name of featurizer')
    parser.add_argument('--time_budget', type=int, help='Time budget for training in seconds')
    parser.add_argument('--strategy', type=str, help='Search strategy')
    parser.add_argument('--metric', type=str, help='Metric for search strategy')
    parser.add_argument('--experiment_name', type=str, help='Name of experiment')

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    featurizer_name = args.featurizer
    time_budget = args.time_budget
    strategy = args.strategy
    metric = args.metric
    experiment_name = args.experiment_name

    x_train, x_test, y_train, y_test = prepare_data(dataset_dir)
    if featurizer_name is not None:
        x_train, x_test = featurize_data(featurizer_name, x_train, x_test)

    nni_predictor = NNIPredictor(x=x_train, y=y_train, time_limit=time_budget, max_epochs=100, test_size=0.2, batch_size=256,
                                 search_strategy=strategy, model_space=[], metric=metric, experiment_name=experiment_name)
    nni_predictor.fit(refit_best=True)
    scores = nni_predictor.score(x_test, y_test)
    print(scores)

    with open(experiment_name + '.txt', 'w') as f:
        f.write(str(scores))

