""" Generate 5000 points in two concentric circles then analyze and plotthe topology using kmapper """
import pandas as pd
import plotly.express as xp

import kmapper as km
from sklearn import datasets


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


columns = list('xy')
kwargs = dict(n_samples=5000, noise=0.03, factor=0.3)
df, labels = datasets.make_circles(**kwargs)
df = pd.DataFrame(df, columns=columns)
fig = xp.scatter(df, x='x', y='y')
html_raw = fig.to_html()
with open('plots/two_concentric_circles_raw.html', 'w') as fout:
    fout.write(html_raw)


# TODO: plot raw data with colors associated with labels

mapper = km.KeplerMapper(verbose=2)

# Fit to and transform the data
dfk = pd.DataFrame(mapper.fit_transform(df, projection=columns), columns=columns)
fig = xp.scatter(dfk, x='x', y='y')
html_xformed = fig.to_html()
with open('plots/two_concentric_circles_transformed.html', 'w') as fout:
    fout.write(html_xformed)

# TODO: plot the projected data with colors for the labels

# Create dictionary of the graph (network) of nodes, edges and meta-information
graph = mapper.map(dfk, df, cover=km.Cover(n_cubes=10))

# Visualize it
html_mapper = mapper.visualize(
    graph,
    path_html="plots/two_concentric_circles_kmapper.html",
    title=f"sklearn.datasets.make_circles({argify(kwargs)})")
