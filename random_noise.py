import os
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import seaborn  # noqa
import matplotlib.pyplot as plt


from ripser import ripser
import persim

HTML_HEADER = """<!DOCTYPE HTML>
<HTML lang="en">
    <HEAD>
        <META charset="utf-8"/>
    </HEAD>
"""
HTML_FOOTER = """</html>"""


def fig_to_html(fig=None, path='/midata/public/tda.html', header='', footer=''):
    fig = fig or plt.gcf() or plt.figure()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = f"{header}<img src='data:image/png;base64,{encoded}'>{footer}".format(encoded)

    if path:
        path = os.path.join(path, f'{__name__}_plot_to_html.html') if os.path.isdir(path) else path
        with open(path, 'w') as f:
            f.write(html)
    return html


def generate_gmm(num_points=100, num_dimensions=2, means=[0], stds=[1], shape=None):
    """ Gennerate Gaussian Mixture Model data points using the specified means and standard deviations

    >>> df = generate_gmm((100, 5), means=[0, 3, 4, 5], stds=[1, .3, .4, .5])
    >>> df.shape
    (400, 5)
    >>> df.mean()
    """
    shape = num_points if shape is None and isinstance(num_points, tuple) else shape
    if isinstance(shape, (tuple, list)):
        num_points, num_dimensions = shape
    data = []
    for m, s in zip(means, stds):
        data += list(m + s * np.random.randn((num_points, num_dimensions)))
    return pd.DataFrame(data)


def tda_to_html(data, show=False, plot_only=[0, 1, 2], max_dim=2):
    topo = ripser(data, max_dim=2)
    html = HTML_HEADER
    html += '<TABLE>\n'
    html += '  <tr>\n    <th>Parameter</th>\n    <th>Value</th>\n  </tr>\n'
    for k in 'cocycles num_edges r_cover'.split():
        html += f'  <tr>\n    <td>{k}</td>\n    <td>{topo[k]}</td>\n  </tr>\n'
    html += '</TABLE>\n'
    persim.plot_diagrams(topo['dgms'], show=show, plot_only=plot_only or None)
    html += fig_to_html()
    topo['html'] = html
    return topo


if __name__ == '__main__':
    NUM_DIMENSIONS = 2
    df = generate_gmm(100, NUM_DIMENSIONS)
    df = generate_gmm()
    tda_to_html()
