from sklearn.cluster import DBSCAN
import os
import sys
import pandas as pd
import numpy as np
# from descartes import PolygonPatch
import matplotlib.pyplot as plt
# from scipy.spatial import Delaunay
# import alphashape

NOISE = 0
UNASSIGNED = 0
core = -1
edge = -2

def DbSCAN_for_activations(top_pixels, image_size, a):
    n_clusters_ = 15
    # top_pixels_2 = one_dimensional_breaks.jenks_breaks(activations, numberOfBreaks, image_size, n, len(top_pixels),
    #                                                  topquadrant + 1)
    # top_pixels = top_pixels_2
    n_noise_ = len(top_pixels)
    print(len(top_pixels))
    i = 0
    eps = 12
    minpts = 20
    while n_clusters_ > 7 or n_noise_ > len(top_pixels) / 4:
        i += 1
        db = DBSCAN(eps=eps, min_samples=minpts).fit(top_pixels)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        outliers = (1 if -1 in labels else 0)
        n_clusters_ = len(set(labels)) - outliers
        eps += 1
        minpts += 2
        n_noise_ = list(labels).count(-1)
        data = np.array(top_pixels)
        nPoints = len(data)
    print(n_clusters_)

    masks = []
    masks_with_ones = []

    for i in range(n_clusters_):
        x1 = []
        y1 = []
        mask = np.zeros(shape=(1, image_size, image_size, 3))
        mask_with_ones = np.zeros(shape=(1, image_size, image_size, 3))
        points = []
        for j in range(nPoints):
            if db.labels_[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
                index_to_be_set = np.argmax(a[0, image_size - int(data[j, 1]), int(data[j, 0])])
                new_mask = [0, 0, 0]
                new_mask[index_to_be_set] = 1
                mask[0, image_size - int(data[j, 1]), int(data[j, 0])] = new_mask
                mask_with_ones[0, image_size - int(data[j, 1]), int(data[j, 0])] = [1, 1, 1]
                points.append([image_size - int(data[j, 1]), int(data[j, 0])])
        # alpha_shape = alphashape.alphashape(points, 0.2)
        masks.append(mask)
        masks_with_ones.append(mask_with_ones)

    return masks, masks_with_ones