import os
import numpy as np
import pandas as pd
import seaborn as sns
#import plotly.express as px 
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


data = pd.read_csv( '/data.csv')
genre_data = pd.read_csv( '/data_w_genres.csv')
artist_data = pd.read_csv( '/data_by_artist.csv')


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

#!pip install spotipy

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from spotipy.oauth2 import SpotifyOAuth


#export SPOTIPY_CLIENT_ID='ba503ee919b241a19afb3a14415d3095'
#export SPOTIPY_CLIENT_SECRET='f628162109f840269d9bf8197f8f31cd'
client_id="ba503ee919b241a19afb3a14415d3095"
client_secret="secret"

#fr#258cdcea1fd8494cb7a0105a282a3141

#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id, client_secret))

#client_id=os.environ["SPOTIFY_CLIENT_ID"],client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=12):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


import json
import time
from spotify_client_credentials import *

client_credentials_manager = SpotifyClientCredentials(client_id,client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_track_ids(playlist_id):
    music_id_list = []
    playlist = sp.playlist(playlist_id)
    for item in playlist['tracks']['items']:
        music_track = item['track']
        music_id_list.append(music_track['id'])
    return music_id_list

def get_track_data(track_id):
    meta = sp.track(track_id)
    track_details = {"name":meta['name'],"album":meta['album']['name'],"artist":meta['album']['artists'][0]['name'],
                     "release_data":meta['album']['release_date'], "duration_in_mins":round((meta['duration_ms']*.001)/60,2)}
    return track_details

playlist_id = '5796702459ca4d40'
track_ids = get_track_ids(playlist_id)

tracks = []

for i in range(len(track_ids)):
    time.sleep(.5)
    track = get_track_data(track_ids[i])
    tracks.append(track)

with open('spotify_data.json','w') as outfile:
    json.dump(tracks,outfile,indent=4)

f = open('spotify_data.json')

data = json.load(f)
names = []
years = []
count = 0
reccomender_str = ""
for i in data:
    names.append(i['name'])
    years.append(i['release_date'][:4])
    song_dict = dict(name=i['name'],year=i['release_date'][:4])
    #print(song_dict)
    #print(song_dict['name'])#, song_dict)
    recommend_songs([{'name':str(song_dict['name']),'year':int(song_dict['year'])}],spotify_data)
    
    
    #print(names[count],years[count])
#     if(count!=len(track_ids)-1):
#         reccomender_str+="{'name':'"+names[count]+"', 'year':"+years[count]+"},"
#     else:
#         reccomender_str+="{'name':'"+names[count]+"', 'year':"+years[count]+"}"
    
    count+=1
#print(reccomender_str)   
    
f.close()
# recommend_songs([reccomender_str],  data)
# recommend_songs([{'name': 'Dream On', 'year':1980},
#                 {'name': 'Hotel California - 2013 Remaster', 'year': 1976},
#                 {'name': 'Break on Through (To the Other Side)', 'year': 1967},
#                 {'name': 'Your Time Is Gonna Come - Remaster', 'year': 1969},
#                 {'name': 'My Type', 'year': 2014}],  data)