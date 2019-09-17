import pandas as pd
import numpy as np
import pickle
import ast
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# path to folder that contains the raw data files
data_file_path = '/Users/ishareef7/Springboard/Capstone2/data/raw/fma_metadata'

# paths to data files
features_path = os.path.join(data_file_path, 'features.csv')
echonest_path = os.path.join(data_file_path, 'echonest.csv')
tracks_path = os.path.join(data_file_path, 'tracks.csv')
genres_path = os.path.join(data_file_path, 'genres.csv')

# path to folder that contains the raw data files
crnn_predictions_path = '/Users/ishareef7/Springboard/Capstone2/src/data'

# paths to data files
train_predictions_path = os.path.join(crnn_predictions_path, 'train_predictions.pickle')
val_predictions_path = os.path.join(crnn_predictions_path, 'val_predictions.pickle')
test_predictions_path = os.path.join(crnn_predictions_path, 'test_predictions.pickle')


def get_features():
    """Return pandas dataframe from features.csv"""

    # read csv file
    features = pd.read_csv(features_path, header=[0, 1, 2], index_col=0)

    return features


def get_echonest():
    """Return pandas dataframe from echonest.csv"""

    # read csv file
    echonest = pd.read_csv(echonest_path, skiprows=[0], header=[0, 1], index_col=0)

    return echonest


def get_tracks():
    """Return pandas dataframe from tracks.csv"""

    # read csv file
    tracks = pd.read_csv(tracks_path, header=[0, 1], index_col=0)

    return tracks


def get_genres(file_path=features_path):
    """Return pandas dataframe from features.csv"""

    # read csv file
    genres = pd.read_csv(genres_path, header=[0], index_col=0)

    return genres

def get_crnn_train_predictions():
    """Return pandas dataframe of training predictions from crnn model"""

    #load pickeled dataframe
    train_predictions = pd.read_pickle(train_predictions_path)

    return train_predictions


def get_crnn_val_predictions():
    """Return pandas dataframe of validation predictions from crnn model"""

    # load pickeled dataframe
    val_predictions = pd.read_pickle(val_predictions_path)

    return val_predictions

def get_crnn_test_predictions():
    """Return pandas dataframe of test predictions from crnn model"""

    # load pickeled dataframe
    test_predictions = pd.read_pickle(test_predictions_path)

    return test_predictions

def get_dataset(subset = 'small', echo = False):
    """Wrangle data for dataset containing echonest features and return dataframe"""
    # load dataframes
    features = get_features()

    tracks = get_tracks()
    genres = get_genres()

    # Correct lists that were loaded into the dataset as strings
    tracks['track', 'genres'] = tracks['track', 'genres'].apply(lambda x: ast.literal_eval(x))

    # Get relevant colmuns from tracks
    idx = pd.IndexSlice
    track_info = tracks.loc[:, idx[['set', 'track'], ['duration', 'split', 'subset', 'genre_top', 'genres']]].copy()
    track_info.columns = track_info.columns.droplevel(0)

    # Create masks used to access rows of track_info where 'genre_top' is null and there is a genre in 'genres'
    null_mask = track_info['genre_top'].isnull().copy()
    null_genres = track_info[null_mask]
    empty_mask = [len(x) > 0 for x in null_genres['genres']]
    null_genres = null_genres[empty_mask]

    # Handle rows were 'genre_top' is blank and infer genre from the first entry in 'genres'
    #genre_dict = dict(genres['title'])
    #null_genres['genre_top'] = null_genres['genres'].apply(lambda x: x[0])
    #top_genres = [genre_dict[x] for x in null_genres['genre_top'].values]
    #track_info.loc[null_genres.index, 'genre_top'] = top_genres

    # Map all rows to their top level genre
    top_genre_dict = dict(zip(genres['title'], genres['top_level']))
    track_info['genre_top'] = track_info['genre_top'].map(top_genre_dict)

    if echo == True:
        # Get numerical echnoest features
        echonest = get_echonest()
        echonest_features = echonest.loc[:, idx[['audio_features', 'temporal_features'], :]]

        # Flatten Indexes
        features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
        echonest_features.columns = ['_'.join(col).rstrip('_') for col in echonest_features.columns.values]

        dataset = features.join(echonest_features, how='inner').join(track_info, how='inner')
        dataset['genre_top'] = dataset['genre_top'].astype('int64')
        dataset= dataset.loc[dataset.subset == subset, :]

    else:
        # Flatten Indexes
        features.columns = ['_'.join(col).strip('_') for col in features.columns.values]

        dataset = features.join(track_info, how = 'inner').dropna()
        dataset['genre_top'] = dataset['genre_top'].astype('int64')

        dataset = dataset.loc[dataset.subset == subset, :]

    return dataset


def get_processed_dataset(subset = 'small', echo = False, validation = False, crnn_predictions = True):
    """Extract and scale features of inputed dataframe. Encode output labels and split into training and test sets"""

    excluded_tracks = [98565, 98567, 98569, 99134, 108925, 133297]

    df = get_dataset(subset, echo).drop(excluded_tracks)
    features = df.iloc[:, :-5].columns
    y = df['genre_top']

    if validation == True:
        train = df['split'] == 'training'
        test = df['split'] == 'test'
        val = df['split'] == 'validation'

        X_train = df.loc[train, features]
        X_test = df.loc[test, features]
        X_val = df.loc[val, features]

        if crnn_predictions:
            train_predictions = get_crnn_train_predictions()
            val_predictions = get_crnn_val_predictions()
            test_predictions = get_crnn_test_predictions()

            X_train = X_train.join(train_predictions).dropna()
            X_val = X_val.join(val_predictions).dropna()
            X_test = X_test.join(test_predictions).dropna()

            # Create scaler instance and transform train and test features
            scaler = StandardScaler(copy=False)
            transformer = ColumnTransformer([('scaler', scaler,X_train.iloc[:, :-8].columns)],
                                            remainder = 'passthrough', sparse_threshold = 0)
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)
            X_val = transformer.transform(X_val)


        else:
            # Create scaler instance and transform train and test features
            scaler = StandardScaler(copy=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_val = scaler.transform(X_val)

        # Create Encoder instance and encode train and test labels
        enc = LabelEncoder()
        y_train = enc.fit_transform(y[train])
        y_test = enc.transform(y[test])
        y_val = enc.transform(y[val])

        return X_train, X_test, X_val, y_train, y_test, y_val
    else:

        train = (df['split'] == 'training') | (df['split'] == 'validation')
        test = df['split'] == 'test'

        X_train = df.loc[train, features]
        X_test = df.loc[test, features]

        if crnn_predictions:
            train_predictions = get_crnn_train_predictions()
            val_predictions = get_crnn_val_predictions()
            test_predictions = get_crnn_test_predictions()

            train_val_predictions = pd.concat([train_predictions, val_predictions])

            X_train = X_train.join(train_val_predictions).dropna()
            X_test = X_test.join(test_predictions).dropna()

            # Create scaler instance and transform train and test features
            scaler = StandardScaler(copy=False)
            transformer = ColumnTransformer([('scaler', scaler, X_train.iloc[:, :-8].columns)],
                                            remainder='passthrough', sparse_threshold=0)
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)

        else:
            # Create scaler instance and transform train and test features
            scaler = StandardScaler(copy=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Create Encoder instance and encode train and test labels
        enc = LabelEncoder()
        y_train = enc.fit_transform(y[train])
        y_test = enc.transform(y[test])

        return X_train, X_test, y_train, y_test