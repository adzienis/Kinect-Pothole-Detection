"""
Kinect Pothole Detection

Arthur Dzieniszewski &
William Mitchell

Final project created for CMU's spring 15-418,
Parallel Computer Architecture & Programming course.

Primary goal of the project was to investigate the usage of MPI,
specifically OpenMPI, in the context of a real world setting. Our real
world setting is composed by trying to identify potholes given the
depth image from a Kinect V1 (Xbox 360). Originally,
the goal of the project was to utilize Raspberry Pi's in a scatter-gather
pattern to speed up our implementation of k-means using Lloyd's Method.
Logistically, we found this difficult to implement since we had to face issues
outside of the project scope (power, wire connections, ethernet controllers).
Instead, we focused on the analysis of an MPI implementation on one of our laptops
utilizing an Intel Core i5-4200M, with 2 physical cores, each hyperthreaded. We ran
tests using our identical implementation against other k-means implementations, with
results included in our report.

Our implementation is written in C, with a wrapper in Python for us to communicate
with the workers spinning.

Our implementation follows the following format:
"""

import os
import sys
import struct

import freenect
import matplotlib.pyplot as plt
import frame_convert
import signal
import numpy as np
import time
import scipy.cluster as sp
import scipy.signal
import matplotlib as mpl
import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import filters
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from scipy.cluster.vq import vq, kmeans, whiten

from sklearn.metrics import silhouette_score
from mpi4py import MPI
from scipy.signal import medfilt

from sklearn.cluster import DBSCAN

keep_running = True

"""
Macros to define the tags for the corresponding values when
we send over to the C workers
"""
CENTROIDS = 6
ALL_LABELS = 7
NUM_DATA = -1

"""
Wrappers around functions included in libfreenect
to pull data from the Kinect. get_depth and get_raw_depth
differ in that get_depth has values scaled down to 255,
while get_raw_depth consists of values from 0-2023. get_depth
is therefore non-injective.
"""
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


"""

"""
def Kmeans(x, num_clusters, features):
    len_arr = (x.shape[0])
    stds = (np.std(x, axis=0))
    x = x/stds

    x = x.flatten()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    new_comm = comm.Split(0, 0)

    start = time.time()

    comm.Isend(struct.pack("<i", len_arr), 1, 0)
    comm.Isend(struct.pack("<i", num_clusters), 1, 1)
    comm.Isend(struct.pack("<i", features), 1, 2)
    comm.Send(struct.pack("%sf" % len_arr*features, *x), 1, 3)

    centroids = 10.*np.arange(num_clusters*features,dtype=np.float64)
    all_labels = 10.*np.arange(len_arr,dtype=np.int32)

    comm.Recv(centroids, source=1, tag=CENTROIDS)
    comm.Recv(all_labels, source=1, tag=ALL_LABELS)

    centroids = np.asarray(struct.unpack_from("%sf" % num_clusters*features, centroids.data))
    all_labels = np.asarray(struct.unpack_from("%si" % len_arr, all_labels.data))

    return (centroids, all_labels, stds)


plt.ion()
print('Press Ctrl-C in terminal to stop')
signal.signal(signal.SIGINT, handler)

pca = PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

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

    if reshaper_where.size <=  10:
        continue

    std = np.std(reshaper_where, axis=0)

    ax2.imshow(depth)

    max_score = -4000

    centroids = None

    indices = np.random.random_integers(0, reshaper_where.shape[0] - 1, 200)

    stds = np.std(reshaper_where, axis=0)

    whitened = whiten(reshaper_where)

    start = time.time()
    for i in range(2, 5):
        np.random.shuffle(reshaper_where)
        (roids, labels,stds) = Kmeans(reshaper_where, i, 2)
        #(roids, labels) = kmeans2(whitened, i, iter=1)

        #kmeans = KMeans(n_clusters=i, n_init=1, algorithm="elkan", tol=.0001).fit(reshaper_where)

        #roids = kmeans.cluster_centers_

        #labels = kmeans.labels_

        try:
            score = silhouette_score(reshaper_where[indices, :], labels[indices])

            if score > max_score:
                max_score = score
                centroids = roids.reshape(-1, 2)*stds
        except:
            continue

    ax3.imshow(get_video())

    print(centroids.shape)

    ax1.scatter(centroids[: , 1], centroids[:, 0])

    plt.pause(.001)
    plt.show()

    ax1.clear()
    ax2.clear()
    ax3.clear()
