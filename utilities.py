from scipy.io import loadmat
from scipy.cluster.vq import vq, kmeans
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from random import shuffle, randint


class Utilities:
    def __init__(self):
        self.data, self.gt, self.data_a = self.mat_to_array()  # (83, 86, 204) ;  (83, 86)
        # for i in range(83):
        #     for j in range(86):
        #         self.gt[i][j] = self.gt[i][j] / 99
        # v, k = kmeans(self.data, 6)
        # at this stage i have to create patches to get a
        # plt.imshow(self.gt, cmap='gray', vmin=0, vmax=14)  # 14/255
        # plt.show()
        x, y, z = self.data_a.shape
        self.data_2d = self.data_a.reshape(x * y, z)
        kmeans_cluster = cluster.KMeans(n_clusters=6)
        kmeans_cluster.fit(self.data_2d)
        cluster_centers = kmeans_cluster.cluster_centers_
        # print(cluster_centers)
        cluster_labels = kmeans_cluster.labels_
        # print(cluster_labels)
        
        # plt.figure(figsize=(83, 86))
        # plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))
        # plt.show()

    def mat_to_array(self):
        data_mat = loadmat('sets/SalinasA_corrected.mat')
        data_a = loadmat('sets/SalinasA.mat')
        gt_mat = loadmat('sets/SalinasA_gt.mat')

        data = data_mat['salinasA_corrected'].astype(np.float32)
        data_a = data_a['salinasA'].astype(np.float32)
        gt = gt_mat['salinasA_gt'].astype(np.float32)

        for i in range(data.shape[-1]):
            data[:, :, i] = (data[:, :, i] - np.mean(data[:, :, i])) / np.std(data[:, :, i])

        for i in range(data_a.shape[-1]):
            data_a[:, :, i] = (data_a[:, :, i] - np.mean(data_a[:, :, i])) / np.std(data_a[:, :, i])

        return data, gt, data_a
