""" Load Sri's data, plot, compute homology, visualize """
import os
import pandas as pd
import numpy as np
import plotly.express as xp

import kmapper as km
from sklearn import datasets
import sklearn.ensemble
import sklearn


N_CUBES = 8
PERC_OVERLAP = .2

def argify(kwarg_dict=None, **kwargs):
    """ Return a string expression equivalent of the ** operator on a dict (k=v arguments)

    >>> argify(**{'hello': 2, 'bye': []})
    'hello=2, bye=[]'
    >>> argify({'hello': 42, 'bye': {}})
    'hello=42, bye={}'
    """
    kwarg_dict = kwarg_dict or {}
    kwarg_dict.update(kwargs)
    return ', '.join(f'{k}={v}' for k, v in kwarg_dict.items())


data_name = 'healthcare'


pd.options.display.max_columns = 400
pd.options.display.width = 180

# from constants import DATA_DIR
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'v2')

disc = pd.read_csv(os.path.join(DATA_DIR, 'discharges.csv'))
disc = disc.dropna(thresh=disc.shape[1] - 10)
disc = disc.dropna(axis=1)
disc = disc[disc.columns[4:]]  # ignore discharge day, mo, year, and discharge_code

visit = pd.read_csv(os.path.join(DATA_DIR, 'visits.csv'), usecols=[0, 1, 2])

print(disc.describe(include='all'))
disc_describe = disc.describe()
numerical_columns = disc_describe.columns
disc = disc[numerical_columns]
fig = xp.scatter_matrix(disc[disc.columns[:8]])
html_raw = fig.to_html()
with open(f'plots/{data_name}_raw.html', 'w') as fout:
    fout.write(html_raw)

visit_describe = visit.describe(include='all')
visit_columns = visit_describe.columns


pca = sklearn.decomposition.PCA(n_components=3)
lens = pd.DataFrame(pca.fit_transform(disc))
isofor = sklearn.ensemble.IsolationForest()
lens['lens_isolation_forest'] = isofor.fit_predict(disc)

mapper = km.KeplerMapper(verbose=2)
lens['l2norm'] = mapper.fit_transform(disc, projection='l2norm')
lens_columns = lens.columns
# lens['color'] = df['color']
fig = xp.scatter_matrix(lens, dimensions=lens_columns) #, color='color')
html_xformed = fig.to_html()
with open(f'plots/{data_name}_lensed.html', 'w') as fout:
    fout.write(html_xformed)



# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(
    lens[lens_columns],
    disc,
    cover=km.Cover(n_cubes=N_CUBES, perc_overlap=PERC_OVERLAP),
    clusterer=sklearn.cluster.KMeans(n_clusters=2))
# Visualize the network in interactive force-directed d3 graph
html_mapper = mapper.visualize(
    graph,
    path_html=f'plots/{data_name}_kmapper.html',
    title=f"KeplerMapper({data_name})")
