""" Generate 5000 points in two nested 2D moons and synthesize polynomials for 4 more dimensions to create troughs/saddles """
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


data_name = 'make_moons'

columns = [f'x{i}' for i in range(6)]
selected = columns[:4]
data_kwargs = dict(n_samples=5000, noise=0.04)
df, labels = datasets.make_moons(**data_kwargs)
df = pd.DataFrame(df, columns=columns[:2])
print(df.shape)
df[columns[2]] = .9 * df[columns[0]].values + .1 * np.random.rand(df.shape[0])
df[columns[3]] = .75 * df[columns[1]].values ** 2 + .25 * np.random.rand(len(df))
df[columns[4]] = .9 * df[columns[0]].values * df[columns[1]].values + .1 * np.random.rand(len(df))
df[columns[5]] = .2 * df[columns[0]].values - .3 * df[columns[1]].values - .4 * df[columns[3]].values + .2 * np.random.randn(len(df))
df['color'] = np.array(['r'] * len(labels))
df['color'][labels.astype(bool)] = 'b'
print(df.describe(include='all'))
fig = xp.scatter_matrix(df, dimensions=columns, color='color')
html_raw = fig.to_html()
with open(f'plots/{data_name}_raw.html', 'w') as fout:
    fout.write(html_raw)


# TODO: plot raw data with colors associated with labels

mapper = km.KeplerMapper(verbose=3)

X = np.array(df[columns])
# create a 1-d Lens based on the 2-norm distance between points
lens = pd.DataFrame()
isofor = sklearn.ensemble.IsolationForest()
lens['lens_isolation_forest'] = isofor.fit_predict(X)
lens['lens_2norm'] = mapper.fit_transform(X, projection='l2norm')

pca = sklearn.decomposition.PCA(n_components=2)
lens_pca = pd.DataFrame(pca.fit_transform(X))
for i, c in enumerate(lens_pca.columns):
    lens[f'pca{i}'] = lens_pca[c]


lens_columns = lens.columns
# lens = create_lens(df[columns])
lens['color'] = df['color']

fig = xp.scatter_matrix(lens, dimensions=lens_columns, color='color')
html_xformed = fig.to_html()
with open(f'plots/{data_name}_lensed.html', 'w') as fout:
    fout.write(html_xformed)

# TODO: plot the projected data with colors for the labels

# Create dictionary of the graph (network) of nodes, edges and meta-information
graph = mapper.map(
    lens[lens_columns],
    df[columns],
    cover=km.Cover(n_cubes=N_CUBES, perc_overlap=PERC_OVERLAP),
    clusterer=sklearn.cluster.KMeans(n_clusters=2))

# Visualize it
html_mapper = mapper.visualize(
    graph,
    path_html=f'plots/{data_name}_kmapper.html',
    title=f"sklearn..{data_name}({argify(data_kwargs)})",
    custom_tooltips=df['color'])
