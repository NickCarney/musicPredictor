import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
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

