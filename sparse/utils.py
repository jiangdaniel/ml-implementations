from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def plot(means, sigmas, values, shape=None, axes=None, flip_y=None, alpha_global=1.0):
    """
    From https://github.com/MaestroGraph/sparse-hyper/
    :param means:
    :param sigmas:
    :param values:
    :param shape:
    :param axes:
    :param flip_y: If not None, interpreted as the max y value. y values in the scatterplot are
            flipped so that the max is equal to zero and vice versa.
    :return:
    """

    b, n, d = means.size()

    means = means.data[0, :,:].cpu().numpy()
    sigmas = sigmas.data[0, :].cpu().numpy()
    values = values.tanh().data[0, :].cpu().numpy()

    if flip_y is not None:
        means[:, 0] = flip_y - means[:, 0]

    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = mpl.cm.RdYlBu
    map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if axes is None:
        axes = plt.gca()

    colors = []
    for i in range(n):
        color = map.to_rgba(values[i])

        alpha = min(0.8, max(0.05, ((sigmas[i, 0] * sigmas[i, 0])+1.0)**-2)) * alpha_global
        axes.add_patch(Ellipse((means[i, 1], means[i, 0]), width=sigmas[i,1], height=sigmas[i,0], color=color, alpha=alpha, linewidth=0))
        colors.append(color)

    axes.scatter(means[:, 1], means[:, 0], s=5, c=colors, zorder=100, linewidth=0, edgecolor='k', alpha=alpha_global)

    if shape is not None:

        m = max(shape)
        step = 1 if m < 100 else m//25

        # gray points for the integer index tuples
        x, y = np.mgrid[0:shape[0]:step, 0:shape[1]:step]
        axes.scatter(x.ravel(),  y.ravel(), c='k', s=5, marker='D', zorder=-100, linewidth=0, alpha=0.1* alpha_global)

    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
