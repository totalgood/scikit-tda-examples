import os
import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from ripser import ripser
from persim import plot_diagrams

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8"/>
    </head>
"""

HTML_FOOTER = """
</html>"""


def plot_to_html(fig=None, path='/midata/public/tda.html'):
    fig = fig or plt.gcf() or plt.figure()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue())
    html = f'{HTML_HEADER}<img src=\'data:image/png;base64,{encoded}\'>{HTML_FOOTER}'.format(encoded) + 'Some more html'

    if path:
        path = path if os.path.isfile(path) else os.path.join(path, '{__name__}_plot_to_html.html')
        with open(path,'w') as f:
            f.write(html)
    return html


def random_noise_tda(num_points=100, num_dimensions=2, shape=None, show=False):
    if isinstance(shape, (tuple, list)):
        num_points, num_dimensions = shape
    data = np.random.random((num_points, num_dimensions))
    diagrams = ripser(data)['dgms']
    plot_diagrams(diagrams, show=show)
    return data, diagrams

if __name__ == '__main__':
    data, diagrams = random_noise_tda()
    plot_to_html()
