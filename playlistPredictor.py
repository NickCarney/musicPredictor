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

data = pd.read_csv( 'data/data.csv')
genre_data = pd.read_csv( 'data/data_w_genres.csv')
artist_data = pd.read_csv( 'data/data_by_artist.csv')

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from spotipy.oauth2 import SpotifyOAuth


#export SPOTIPY_CLIENT_ID='ba503ee919b241a19afb3a14415d3095'
#export SPOTIPY_CLIENT_SECRET='f628162109f840269d9bf8197f8f31cd'
client_id="ba503ee919b241a19afb3a14415d3095"
client_secret = "54015e1aedf0406f960b102633ca6809"

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
    
    metadata_cols = ['name', 'year', 'artists','id']
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
#from spotify_client_credentials import *

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
                     "release_date":meta['album']['release_date'], "duration_in_mins":round((meta['duration_ms']*.001)/60,2)}
    return track_details

playlist_id = '42ukc3Z21ZLWO4F480YEbH'
playlist_name = "''"+sp.playlist(playlist_id)['name']+"''"
track_ids = get_track_ids(playlist_id)

tracks = []

for i in range(len(track_ids)):
    time.sleep(.5)
    track = get_track_data(track_ids[i])
    tracks.append(track)

with open('spotify_data.json','w') as outfile:
    json.dump(tracks,outfile,indent=4)

f = open('spotify_data.json')

playlist_data = json.load(f)
names = []
years = []
count = 0
reccomender_str = ""
dict_list = []
for i in range(len(playlist_data)):
    name = playlist_data[i]['name']
    year = int(playlist_data[i]['release_date'].split('-')[0])
    #print(i,name,year)
    names.append(name)
    years.append(year)
    song_dict = dict(name=names[i],year=years[i])
    dict_list.append(song_dict)
f.close()
recommended_songs = recommend_songs(dict_list , data, len(playlist_data))
     
import spotipy.util as util
user = '31t2mek3n4j35zwlcpdh77bgfce4'#my username

#export SPOTIPY_CLIENT_ID='ba503ee919b241a19afb3a14415d3095'
#export SPOTIPY_CLIENT_SECRET='258cdcea1fd8494cb7a0105a282a3141'

scope = 'playlist-modify-public'
token = util.prompt_for_user_token(user,
                           scope,
                           client_id='ba503ee919b241a19afb3a14415d3095',
                           client_secret='54015e1aedf0406f960b102633ca6809',
                           redirect_uri='https://example.com/callback')
sp2 = spotipy.Spotify(auth=token)
sp2.trace = False
new_playlist = sp2.user_playlist_create(user, playlist_name+" PREDICTED", public=True, description='made based on machine learning predictions using the playlist'+playlist_name)

songs_to_add = []
for i in range(len(recommended_songs)):
    songs_to_add.append(recommended_songs[i]['id'])

repeat = False
for i in range(len(recommended_songs)):
    for i in range(len(sp.playlist(new_playlist['id'])['tracks']['items'])):
        if(sp.playlist(new_playlist['id'])['tracks']['items'][i]['track']['id']==recommended_songs[i]['id']):
            repeat=True
    if(repeat == False):
        results = sp2.user_playlist_add_tracks(user, new_playlist['id'], songs_to_add)

