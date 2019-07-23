import pandas as pd
import numpy as np
import ast
import os

# path to folder that contains the raw data files
data_file_path = '/Users/ishareef7/Springboard/Capstone2/data/raw/fma_metadata'

# paths to data files
features_path = os.path.join(data_file_path, 'features.csv')
echonest_path = os.path.join(data_file_path, 'echonest.csv')
tracks_path = os.path.join(data_file_path, 'tracks.csv')
genres_path = os.path.join(data_file_path, 'genres.csv')


def get_features(file_path = features_path):
    """Return pandas dataframe from features.csv"""

    # read csv file
    features = pd.read_csv(features_path, header=[0, 1, 2], index_col=0)

    return features


def get_echonest(file_path = echonest_path):
    """Return pandas dataframe from echonest.csv"""

    # read csv file
    echonest = pd.read_csv(echonest_path, skiprows=[0], header=[0, 1], index_col=0)

    return echonest


def get_tracks(file_path = tracks_path):
    """Return pandas dataframe from tracks.csv"""

    # read csv file
    tracks = pd.read_csv(tracks_path, header=[0, 1], index_col=0)

    return tracks

def get_genres(file_path = features_path):
    """Return pandas dataframe from features.csv"""

    # read csv file
    genres = pd.read_csv(genres_path, header=[0], index_col=0)

    return genres

#load dataframes
features = get_features()
echonest = get_echonest()
tracks = get_tracks()
genres = get_genres()

#Correct lists that were loded into the dataset as strings
tracks['track','genres'] = tracks['track','genres'].apply(lambda x: ast.literal_eval(x))

#Get relevant colmuns from tracks
idx = pd.IndexSlice
track_info = tracks.loc[:, idx[['set','track'], ['duration','split', 'subset','genre_top','genres']]].copy()
track_info.columns = track_info.columns.droplevel(0)

#Create masks used to access rows of track_info where 'genre_top' is null and there is a genre in 'genres'
null_mask = track_info['genre_top'].isnull().copy()
null_genres = track_info[null_mask]
empty_mask = [len(x)>0 for x in  null_genres['genres']]
null_genres = null_genres[empty_mask]

#Handle rows were 'genre_top' is blannk and infer genre from the first entry in 'genres'
genre_dict = dict(genres['title'])
null_genres['genre_top'] = null_genres['genres'].apply(lambda x: x[0])
top_genres = [ genre_dict[x] for x in null_genres['genre_top'].values]
track_info.loc[null_genres.index, 'genre_top']  = top_genres

#Map all rows to their top level genre
top_genre_dict = dict(zip(genres['title'], genres['top_level']))
track_info['genre_top'] = track_info['genre_top'].map(top_genre_dict)

#Get numerical echnoest features
echonest_features = echonest.loc[:, idx[['audio_features', 'temporal_features'],:]]

#Flatten Indexes
features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
echonest_features.columns = ['_'.join(col).rstrip('_') for col in echonest_features.columns.values]


dataset_echonest = features.join(echonest_features, how = 'inner').join(track_info, how = 'inner')
dataset_no_echnoest = features.join(track_info, how = 'inner')

def get_dataset_echonest():
    """Return dataset that contains collumns for echonest features this dataset only contains 13,129 rows """
    return dataset_echonest

def get_dataset():
    """ Return dataset without echonest features this dataset contains 106,574 rows"""
    return dataset_no_echnoest