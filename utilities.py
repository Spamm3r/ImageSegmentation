from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from sklearn import cluster, mixture


class Utilities:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.data, self.gt, self.data_a = self.mat_to_array()  # (83, 86, 204) ;  (83, 86)
        # kmeans = self.kmeans_for_each_pixel(self.data, self.n_classes)
        kmeans = self.kmeans_custom(self.data, self.n_classes, 1)
        self.pca(kmeans[0])

    def kmeans_custom(self, data, n_classes, columns):
        x, y, z = data.shape
        clustered = np.zeros((x, y, z))
        for i in range(columns):
            for j in range(y):
                clustered[i][j] = cluster.KMeans(n_clusters=n_classes).fit(self.data[i][j].reshape(-1, 1)).labels_
        return clustered

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


    def pca(self, data):
        m = mean(data)
        c = data - m
        v = np.cov(c.T)
        values, vectors = np.linalg.eig(v)
        p = vectors.T.dot(c.T)
        # print(p.T)
        plt.imshow(np.real(p.T))
        plt.show()
