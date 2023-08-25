import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.decomposition import PCA


def Preprocess(data):
    # data preprocessing, including the removal of genes expressed in less than 95% of cells and logarithm
    X = np.array(data)
    print('raw shape:',X.shape)
    X = X.T
    cell = []
    for i in range(len(X)):
        cell.append(len(np.argwhere(X[i] > 0)))
    cell = np.array(cell)
    t = int(0.05 * X.shape[1])
    index = np.argwhere(cell > t)
    index = index.reshape(len(index))
    X = X[index]
    X = np.transpose(X)
    print("after preprocessing", X.shape)
    return X

def normalization(X):
    # data normalization
    X =np.array(X)
    # print(X.shape)
    for i in range(len(X)):
        X[i] = X[i] / sum(X[i]) * 100000
    X = np.log2(X + 1)
    return X

def euclidean_dist(x, y=None):
    """
    Compute all pairwise distances between vectors in X and Y matrices.
    :param x: numpy array, with size of (d, m)
    :param y: numpy array, with size of (d, n)
    :return: EDM:   numpy array, with size of (m,n).
                    Each entry in EDM_{i,j} represents the distance between row i in x and row j in y.
    """
    if y is None:
        y = x

    # calculate Gram matrices
    G_x = np.matmul(x.T, x)
    G_y = np.matmul(y.T, y)

    # convert diagonal matrix into column vector
    diag_Gx = np.reshape(np.diag(G_x), (-1, 1))
    diag_Gy = np.reshape(np.diag(G_y), (-1, 1))

    # Compute Euclidean distance matrix
    EDM = diag_Gx + diag_Gy.T - 2 * np.matmul(x.T, y)  # broadcasting

    # print('This is matrix EDM: ')
    # print(EDM)
    return EDM

def heat_kernel(x, t):
    D2 = euclidean_dist(x)
    heat_kernel = np.exp(-D2 / t)
    return heat_kernel

def getGraph(X, K, method, t=None):
    # print(method)

    if method == 'pearson':
        co_matrix = np.corrcoef(X)
    elif method == 'spearman':
        co_matrix, _ = spearmanr(X.T)
    elif method == 'heat_kernel':
        co_matrix = heat_kernel(X.T, 1)

    # print(co_matrix)

    up_K = np.sort(co_matrix)[:, -K]
    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1
    # print(mat_K)
    mat_K = (mat_K+mat_K.T)/2
    mat_K[mat_K>=0.5] = 1
    W = mat_K*co_matrix
    # pd.DataFrame(mat_K).to_csv('graph.csv', index=False, header=None)
    return mat_K,W

def load_data(data_path, dataset_str, PCA_dim, n_clusters, method, K):
    # Get data
    DATA_PATH = data_path
    data = pd.read_csv(DATA_PATH, header=None, low_memory=False)
    # data = data.iloc[1:, 1:]

    # Preprocess features
    features = Preprocess(data)
    features = normalization(features)

    # Construct graph
    N = features.shape[0]
    avg_N = N // n_clusters
    K = avg_N // 10
    K = min(K, 20)
    K = max(K, 3)
    print('K', K)

    adj,W = getGraph(features, K, 'pearson')

    # feature tranformation
    if features.shape[0] > PCA_dim and features.shape[1] > PCA_dim:
        pca = PCA(n_components=PCA_dim)
        features = pca.fit_transform(features)
    else:
        var = np.var(features, axis=0)
        min_var = np.sort(var)[-1 * PCA_dim]
        features = features.T[var >= min_var].T
        features = features[:, :PCA_dim]
    print('Shape after transformation:', features.shape)

    features = (features - np.mean(features)) / (np.std(features))

    return adj, features


def performance(y,y_pred):
    # Cluster metrics
    NMI = metrics.normalized_mutual_info_score(y, y_pred)
    print("NMI:", round(NMI,3))
    ARI = metrics.adjusted_rand_score(y, y_pred)
    print("ARI:", round(ARI,3))
