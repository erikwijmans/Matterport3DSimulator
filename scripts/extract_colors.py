import sys
sys.path.append('build')
import MatterSim
import time
import math

sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
sim.init()

import glob
import os.path as osp
import numpy as np
from tqdm import tqdm

if osp.exists("all_colors.npy"):
    all_colors = np.load("all_colors.npy")
else:
    all_colors = []
    for name in tqdm(glob.glob("./connectivity/*.json")):
        file_name = osp.split(name)[1]
        scan_id = file_name.split("_")[0]

        sim.newEpisode(scan_id, "", 0, 0)
        objects = sim.get_objects()

        for obj_id in list(objects):
            o = objects[obj_id]
            all_colors.append(np.array([o.color.r, o.color.g, o.color.b]))

    all_colors = np.stack(all_colors)
    np.save("all_colors.npy", all_colors)

import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score


def kmeans(data, n=19):
    centroids, labels, interia = cluster.k_means(
        data, n, n_init=int(1e3), max_iter=int(1e6), precompute_distances=True)

    return centroids, labels, silhouette_score(data, labels)


from sklearn.mixture import GaussianMixture


def gmm(data, n=19):
    gmm = GaussianMixture(n, n_init=int(1e3), max_iter=int(1e6))
    gmm.fit(data)

    labels = gmm.predict(data)

    return gmm.means_, gmm.predict(data), silhouette_score(data, labels)


def dbscan(data):
    dbscan = cluster.DBSCAN(min_samples=10, algorithm="brute")
    dbscan.fit(data)

    return dbscan.components_, dbscan.labels_, silhouette_score(
        data, dbscan.labels_)

def clusterer(data):
    #  centroids, _, score = dbscan(data)
    #  print("DBSCAN: {:1.5f}".format(score))
    #  print(centroids)

    centroids, _, score = kmeans(data)
    print("Kmeans: {:1.5f}".format(score))
    print(centroids)

    centroids, _, score = gmm(data)
    print("GMM: {:1.5f}".format(score))
    print(centroids)


print("Clustering in RGB space")
clusterer(all_colors)


rgb2ycbcr = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5],
                      [0.5, -0.419, -0.081]])

print("Clustering in CbCr space")
ycbcr = all_colors.dot(rgb2ycbcr.T)
cbcr = ycbcr[:, 1:]
clusterer(cbcr)
