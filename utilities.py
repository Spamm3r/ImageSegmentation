from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, mixture
import pandas as pd
from sklearn.decomposition import PCA

class Utilities:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.data, self.gt, self.data_a = self.mat_to_array()  # (83, 86, 204) ;  (83, 86)
        # kmeans = self.kmeans_for_each_pixel(self.data, self.n_classes)

    def kmeans_for_each_pixel(self, data, n_classes):
        x, y, z = data.shape
        clustered = np.zeros((x, y, z))
        for i in range(x):
            for j in range(y):
                clustered[i][j] = cluster.KMeans(n_clusters=n_classes).fit(self.data[i][j].reshape(-1, 1)).labels_
        return clustered

    def mat_to_array(self):
        data_mat = loadmat('sets/SalinasA_corrected.mat')
        data_a = loadmat('sets/SalinasA.mat')
        gt_mat = loadmat('sets/SalinasA_gt.mat')

        data = data_mat['salinasA_corrected'].astype(np.float32)
        data_a = data_a['salinasA'].astype(np.float32)
        gt = gt_mat['salinasA_gt'].astype(np.float32)

        for i in range(data.shape[-1]):
            data[:, :, i] = (data[:, :, i] - np.mean(data[:, :, i])) / np.std(data[:, :, i])

        # for i in range(data_a.shape[-1]):
        #     data_a[:, :, i] = (data_a[:, :, i] - np.mean(data_a[:, :, i])) / np.std(data_a[:, :, i])

        return data, gt, data_a
