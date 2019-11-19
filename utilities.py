from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
import math
from sklearn import cluster, mixture, datasets
from sklearn.decomposition import PCA
from fcmeans import FCM
from seaborn import scatterplot as scatter


class Utilities:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.data, self.gt, self.data_a = self.mat_to_array()  # (83, 86, 204) ;  (83, 86)
        kmeans = self.kmeans_for_each_pixel(self.data, self.n_classes)
        # self.fuzzy_c_means(self.data[81], self.n_classes)
        # kmeans = self.kmeans_custom(self.data, self.n_classes, 11)
        # self.pca(kmeans[0])
        # pca = PCA()
        # self.y = list(range(0, 86))
        # pca.fit(kmeans[0][13], self.y)
        # x_new = pca.transform(kmeans[0][13])
        # self.custom_plot(x_new[:, 0:2], pca.components_)
        # plt.show()

    def fuzzy_c_means(self, data, n_classes):
        fcm = FCM(n_clusters=n_classes)
        fcm.fit(data)
        fcm_centers = fcm.centers
        fcm_labels = fcm.u.argmax(axis=1)

        f, axes = plt.subplots(1, 2, figsize=(11, 5))
        scatter(data[:, 0], data[:, 1], ax=axes[0])
        scatter(data[:, 0], data[:, 1], ax=axes[1], hue=fcm_labels)
        scatter(fcm_centers[:, 0], fcm_centers[:, 1], ax=axes[1], marker="s", s=200)
        plt.show()


    def kmeans_custom(self, data, n_classes, columns):
        x, y, z = data.shape
        clustered = np.zeros((x, y, z))
        for i in range(columns):
            for j in range(y):
                clustered[i][j] = cluster.KMeans(n_clusters=n_classes).fit(self.data[i][j].reshape(-1, 1)).labels_
        return clustered

    def custom_plot(self, score, coeff, labels=None):
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]

        plt.scatter(xs, ys, c=self.y)  # without scaling
        for i in range(n):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
            if labels is None:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center',
                         va='center')
            else:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()

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
            data[:, :, i] = (data[:, :, i] - np.mean(data[:, :, i])) / np.std(data[:, :, i])  # Standard score of the x


        # for i in range(data_a.shape[-1]):
        #     data_a[:, :, i] = (data_a[:, :, i] - np.mean(data_a[:, :, i])) / np.std(data_a[:, :, i])

        return data, gt, data_a

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def pca(self, data):
        m = mean(data)
        c = data - m
        v = np.cov(c.T)
        values, vectors = np.linalg.eig(v)
        p = vectors.T.dot(c.T)
        # print(p.T)
        plt.imshow(np.real(p.T))
        plt.show()
