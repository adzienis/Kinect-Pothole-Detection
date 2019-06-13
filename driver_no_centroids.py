#!/usr/bin/env python
import freenect
import matplotlib.pyplot as plt
import frame_convert
import signal
import numpy as np
import time
import scipy.cluster as sp
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import scipy.signal
import matplotlib as mpl

from sklearn.decomposition import PCA
import os
import sys
from sklearn.cluster import KMeans
from skimage import filters

from mpi4py import MPI
from scipy.signal import medfilt

keep_running = True

def get_depth():
    temp = freenect.sync_get_depth()[0]
    return frame_convert.pretty_depth(temp)

def get_raw_depth():
    return freenect.sync_get_depth()[0]

def get_video():
    return freenect.sync_get_video()[0]


def handler(signum, frame):
    """Sets up the kill handler, catches SIGINT"""
    global keep_running
    keep_running = False
    print("killed")
    quit()

plt.ion()
print('Press Ctrl-C in terminal to stop')
signal.signal(signal.SIGINT, handler)

pca = PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=True)

fig= plt.figure(figsize=(24,10))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

while keep_running:
    depth = np.array(get_depth())
    depth_2 = np.array(get_raw_depth())
    depth_2 = medfilt(depth_2.flatten(), 5).reshape(480, 640)

    flat = depth_2.flatten()

    depth_2_clean = np.delete(flat, np.argwhere(flat == 2047))

    positions = np.argwhere(depth_2 >= 0)

    together = np.concatenate((positions,depth_2.reshape(depth_2.size, 1)), axis=1)

    together_clean = np.delete(together, np.argwhere(flat == 2047), axis=0)

    model = pca.fit(together_clean)

    model.components_ = model.components_

    new = np.dot(together_clean, model.components_.T)

    inverse = np.linalg.inv(model.components_.T)

    inverted = np.matmul(new, inverse)
    starts = np.zeros(3)

    std = np.std(new[:, 2])
    avg = np.mean(new[:, 2])

    threshold = np.argwhere(new[:, 2] > avg + std*2)

    extracted = np.take(new, threshold, axis=0)

    extracted = np.matmul(extracted, inverse).astype(int)

    if extracted.size == 0:
        plt.clf()
        continue

    zeros = np.zeros(extracted.shape[0])
    reshaper = np.zeros(depth.size)

    reshaper[(extracted.astype(int)[:, 0, 1])*480 + (extracted.astype(int)[:, 0, 0])] = 255

    reshaper = reshaper.reshape((640,480)).T

    ax1.imshow(reshaper, cmap=plt.cm.viridis, origin="upper")

    reshaper_where = np.argwhere(reshaper > 0)

    if reshaper_where.size <= 10:
        continue

    std = np.std(reshaper_where, axis=0)

    ax2.imshow(depth)

    max_score = -4000

    centroids = None

    indices = np.random.random_integers(0, reshaper_where.shape[0] - 1, 200)

    stds = np.std(reshaper_where, axis=0)

    whitened = whiten(reshaper_where)

    ax3.imshow(get_video())
    plt.pause(.001)
    plt.show()

    ax1.clear()
    ax2.clear()
    ax3.clear()
