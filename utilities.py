from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, decomposition
from spectral import *


class Utilities:
    def __init__(self, n_classes):
        # preparing data
        self.n_classes = n_classes
        self.data_pavia_u = loadmat('sets/PaviaU.mat')
        self.data_kennedy = loadmat('sets/KSC.mat')
        self.data_salinas_a_corrected = loadmat('sets/SalinasA_corrected.mat')
        self.data_salinas_a = loadmat('sets/SalinasA.mat')
        self.data_pines = loadmat('sets/Indian_pines.mat')
        self.data_pines_corrected = loadmat('sets/Indian_pines_corrected.mat')
        self.data_salinas_a_gt = loadmat('sets/SalinasA_gt.mat')
        self.mat_to_array()

        self.pca_then_kmean1(self.data_pines, 9)

    def pca_then_kmean(self, data, k):
        x, y, z = data.shape
        mod = int(z/10)

        print("Calculating PCA")
        pc = principal_components(data)
        pc_reduced = pc.reduce(fraction=0.999)

        img_pc = pc_reduced.transform(data)

        print("Calculating Kmeans")
        pca = np.zeros(shape=(x, y, mod))
        for i in range(x):
            for j in range(y):
                for l in range(mod):
                    pca[i, j, l] = img_pc[i, j, l]

        self.kmeans(pca, x, y, mod)

    def pca_then_kmean1(self, data, k):
        x, y, z = data.shape

        print("Calculating PCA")
        vectors = data.reshape(x * y, z)
        # vectors = np.transpose(vectors)  # matrix of features
        samples = np.shape(vectors)[1]  # basically rows * columns

        # my own implementation
        # mu = np.mean(vectors, axis=1)  # mean
        # for n in range(samples):
        #     vectors[range(z), n] = vectors[(range(z), n)] - mu  # filling with values

        # c = np.cov(vectors)  # cov matrix of features
        # d, v = np.linalg.eig(c)  # eigenvalues of cov matrix of features
        # vec_pca = np.dot(v.T, vectors)  # projecting data into eigenvector

        # sklearn pca
        pca1 = decomposition.PCA(n_components=17)
        vec_pca1 = pca1.fit_transform(vectors)

        # pca = vec_pca1[:, 0]
        # pca_im = np.reshape(pca, (x, y), order='F')
        # plt.figure(figsize=(7, 7))
        # plt.imshow(np.abs(pca_im))
        # plt.show()

        # pca2 = decomposition.PCA(n_components=int(z/10))
        # vec_pca2 = pca2.fit_transform(vectors)

        # fig = plt.figure(13)
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(vec_pca1[range(samples), 0], vec_pca1[range(samples), 1], vec_pca1[range(samples), 2], marker='o')
        # plt.show()

        # Displaying an cov matrix with values after PCA
        # cov_pca = np.cov(vec_pca)
        # d, v = np.linalg.eig(c)
        # d = d.real
        # for r in range(10):
        #     print('{0:5f} {1:5f}'.format(d[r], cov_pca[r, r]))
        # print()
        # for r in range(10):
        #     for c in range(10):
        #         n_val = int(10000 * cov_pca[r, c])
        #         print('{0:5d}'.format(n_val), end=" ")
        #     print('\n')

        # scatter of the vecPCA
        # fig = plt.figure(13)
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(vec_pca[0, range(samples)], vec_pca[1, range(samples)], vec_pca[2, range(samples)], marker='o')
        # plt.show()

        # for coord in range(3):
        #     pca = vec_pca[coord, :]
        #     pca_im = np.reshape(pca, (x, y), order='F')
        #     plt.figure(14 + coord)
        #     plt.imshow(np.abs(pca_im))
        #     plt.colorbar()
        #     plt.show()

        print("Calculating Kmeans")
        self.kmeans1(vec_pca1, x, y, z, k)

    def kmeans1(self, data, x, y, z, n_classes):
        clf = cluster.KMeans(n_classes)
        img = clf.fit(data)
        plt.figure(figsize=(7, 7))
        plt.imshow(img.labels_.reshape((x, y)))
        plt.show()

    def kmeans(self, data, x, y, k):
        (c, v) = kmeans(data, 7, 300)
        plt.imshow(c)
        plt.show()

    def mat_to_array(self):
        self.data_salinas_a_gt = self.data_salinas_a_gt['salinasA_gt'].astype(np.float32)

        self.data_pavia_u = self.data_pavia_u['paviaU'].astype(np.float32)
        self.data_kennedy = self.data_kennedy['KSC'].astype(np.float32)
        self.data_salinas_a_corrected = self.data_salinas_a_corrected['salinasA_corrected'].astype(np.float32)
        self.data_salinas_a = self.data_salinas_a['salinasA'].astype(np.float32)
        self.data_pines = self.data_pines['indian_pines'].astype(np.float32)
        self.data_pines_corrected = self.data_pines_corrected['indian_pines_corrected'].astype(np.float32)

        self.data_pavia_u = self.processing_values(self.data_pavia_u)
        self.data_kennedy = self.processing_values(self.data_kennedy)
        self.data_salinas_a_corrected = self.processing_values(self.data_salinas_a_corrected)
        self.data_salinas_a = self.processing_values(self.data_salinas_a)
        self.data_pines = self.processing_values(self.data_pines)
        self.data_pines_corrected = self.processing_values(self.data_pines_corrected)

    def processing_values(self, data):
        for i in range(data.shape[-1]):
            data[:, :, i] = (data[:, :, i] - np.mean(data[:, :, i])) / np.std(data[:, :, i])
        return data
