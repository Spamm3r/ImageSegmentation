from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from sklearn import cluster, decomposition
from mpl_toolkits.mplot3d import axes3d, Axes3D


class Utilities:
    def __init__(self, n_classes):
        # preparing data
        self.n_classes = n_classes
        self.data, self.gt, self.data_a = self.mat_to_array()  # (83, 86, 204) ;  (83, 86)

        # algorithms
        self.pca_then_kmean(self.data, 24)

    def pca_then_kmean(self, data, components):
        x, y, z = data.shape

        print("Calculating PCA")
        vectors = data.reshape(x*y, z, order="F")
        vectors = np.transpose(vectors)  # matrix of features
        samples = np.shape(vectors)[1]  # basically rows * columns = 7138

        mu = np.mean(vectors, axis=1)  # mean
        for n in range(samples):
            vectors[range(z), n] = vectors[(range(z), n)] - mu  # filling with values

        c = np.cov(vectors)  # cov matrix of features
        d, v = np.linalg.eig(c)  # eigenvalues of cov matrix of features
        vec_pca = np.dot(v.T, vectors)  # projecting data into eigenvector

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

        # for coord in range(1):
        #     pca = vec_pca[coord, :]
        #     pca_im = np.reshape(pca, (x, y), order='F')
        #     plt.figure(14 + coord)
        #     plt.imshow(np.abs(pca_im))
        #     plt.colorbar()
        #     plt.show()

        print("Calculating Kmeans")

        # amount of components: 3 and 20
        # arr1 = self.array_of_components(vec_pca, 6, samples)
        # arr2 = self.array_of_components(vec_pca, 20, samples)

        self.kmeans(self.array_of_components(vec_pca, 6, samples), x, y, 6)
        self.kmeans(self.array_of_components(vec_pca, 20, samples), x, y, 6)

        # arr1 = arr1.T
        # clf = cluster.KMeans(n_clusters=6)
        # labels = clf.fit_predict(arr1)
        # plt.figure(figsize=(12, 12))
        # plt.imshow(labels.reshape((83, 86)), cmap='gray')
        # plt.show()
        #
        # arr2 = arr2.T
        # clf = cluster.KMeans(n_clusters=6)
        # labels = clf.fit_predict(arr2)
        # plt.figure(figsize=(12, 12))
        # plt.imshow(labels.reshape((83, 86)), cmap='gray')
        # plt.show()

        # kmeans2 = np.zeros(shape=(20, samples))

    def array_of_components(self, data, n_components, second_dim):
        arr = np.zeros(shape=(n_components, second_dim))

        for i in range(n_components):
            for j in range(second_dim):
                arr[i][j] = data[i][j]
        return arr

    def kmeans(self, data, x, y, n_classes):
        clf = cluster.KMeans(n_clusters=n_classes)
        labels = clf.fit_predict(data.T)
        plt.figure(figsize=(12, 12))
        plt.imshow(labels.reshape((x, y)), cmap='gray')
        plt.show()

    # def kmeans_for_each_pixel(self, data, n_classes):
    #     x, y = data.shape
    #     clustered = np.zeros((n_classes, y))
    #     for i in range(y):
    #         clustered[range(x), i] = cluster.KMeans(n_clusters=n_classes).fit(data[range(x), i].reshape(-1, 1)).labels_
    #         # kmean = cluster.KMeans(n_clusters=n_classes).fit(data[i][j].reshape(-1, 1))
    #     return clustered

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

        # for i in range(data.shape[-1]):
        #     data[:, :, i] = StandardScaler().fit_transform(data[:, :, i])

        return data, gt, data_a
