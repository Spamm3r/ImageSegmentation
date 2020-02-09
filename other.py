# samples = np.shape(vectors)[1]  # basically rows * columns = 7138
#
# mu = np.mean(vectors, axis=1)  # mean
# for n in range(samples):
#     vectors[range(z), n] = vectors[(range(z), n)] - mu  # filling with values
#
# c = np.cov(vectors)  # cov matrix of features
# d, v = np.linalg.eig(c)  # eigenvalues of cov matrix of features
# vec_pca = np.dot(v.T, vectors)  # projecting data into eigenvector
