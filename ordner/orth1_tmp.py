''' This is the python code for the corresponding matlab file my_orth1'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from calculate_mdl import AutoOrth, _calculate_rotation, _update_centers_and_scatter_matrices, _mdl_costs
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import datasets.real_world_data as datasets
from metrics import MultipleLabelingsConfusionMatrix, ConfusionMatrix, variation_of_information, \
    unsupervised_clustering_accuracy, PairCountingScores
from sklearn.metrics import normalized_mutual_info_score as nmi

try:
    # Old sklearn versions
    from sklearn.cluster._kmeans import _k_init as kpp
except:
    # New sklearn versions
    from sklearn.cluster._kmeans import _kmeans_plusplus as kpp

_ACCEPTED_NUMERICAL_ERROR = 1e-6
_NOISE_SPACE_THRESHOLD = -1e-7


class Results:
    def __init__(self, pred_labels=None, best_mdl_score=None, best_k=None, best_m=None, best_V_proj=None, nmi_list=None):
        """
        Creates a results instance to save informations. Temporary for now
        :param pred_labels: predicted labels
        :param best_mdl_score: lowest MDL score
        :param best_k: number of clusters of the instance with the lowest MDL score
        :param best_m: number of dimensions of the cluster space of the instance with the lowest MDL score
        :param best_V_proj: transformed data_points to the clusterspace
        """
        self.pred_labels = pred_labels
        self.best_mdl = best_mdl_score
        self.best_k = best_k
        self.best_m = best_m
        self.best_V_proj = best_V_proj
        self.nmi_list = nmi_list

class Orth1:
    def __init__(self, n_clusters, V=None, m=None, P=None, input_centers=None, mdl_for_noisespace=True, outliers=False,
                 max_iter=300, max_distance=None, precision=None, random_state=None, debug=True):
        """
                Create new Orth1 instance. Gives the opportunity to use the fit() method to cluster a dataset.
                :param n_clusters: list containing number of clusters for each subspace_nr
                :param V: orthogonal rotation matrix (optional)
                :param m: list containing number of dimensionalities for each subspace_nr (optional)
                :param P: list containing projections for each subspace_nr (optional)
                :param input_centers: list containing the cluster centers for each subspace_nr (optional)
                :param mdl_for_noisespace: boolean defining if MDL should be used to identify noise space dimensions (default: False)
                :param outliers: boolean defining if outliers should be identified (default: False)
                :param max_iter: maximum number of iterations for the Orth1 algorithm (default: 300)
                :param random_state: use a fixed random state to get a repeatable solution (optional)
                """
        # Fixed attributes
        self.input_n_clusters = n_clusters.copy()
        self.max_iter = max_iter
        self.random_state = random_state
        self.mdl_for_noisespace = mdl_for_noisespace
        self.outliers = outliers
        self.max_distance = max_distance
        self.precision = precision
        self.debug = debug
        # Variables
        self.n_clusters = n_clusters
        self.input_centers = input_centers
        self.V = V
        self.m = m
        self.P = P


    def fit(self, X, y=None):
        """
        Cluster the input dataset with the Orth1 algorithm. Saves the labels, centers, V, m, P and scatter matrices
        in the Orth1 object.
        :param X: input data
        :return: the Orth1 object
        """
        labels, centers, V, m, n_clusters, scatter_matrices = myOrth1(X, self.n_clusters, self.random_state)

        # Update class variables
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.V = V
        self.m = m
        self.n_clusters = n_clusters
        self.scatter_matrices_ = scatter_matrices
        return self

    def transform_full_space(self, X):
        """
        Transform the input dataset with the orthogonal rotation matrix V from the Nr-Kmeans object.
        :param X: input data
        :return: the rotated dataset
        """
        return np.matmul(X, self.V)

    def transform_subspace(self, X, subspace_index):
        """
        Transform the input dataset with the orthogonal rotation matrix V projected onto a special subspace_nr.
        :param X: input data
        :param subspace_index: index of the subspace_nr
        :return: the rotated dataset
        """
        # cluster_space_V = self.V[:, self.P[subspace_index]]
        cluster_space_V = self.V[:, :self.m[subspace_index]]
        return np.matmul(X, cluster_space_V)

    def have_subspaces_been_lost(self):
        """
        Check whether subspaces have been lost during Nr-Kmeans execution.
        :return: True if at least one subspace_nr has been lost
        """
        return len(self.n_clusters) != len(self.input_n_clusters)

    def have_clusters_been_lost(self):
        """
        Check whether clusters within any subspace have been lost during Nr-Kmeans execution.
        Will also return true if subspaces have been lost (check have_subspaces_been_lost())
        :return: True if at least one cluster has been lost
        """
        return not np.array_equal(self.input_n_clusters, self.n_clusters)

    def get_cluster_count_of_changed_subspaces(self):
        """
        Get the Number of clusters of the changed subspaces. If no subspace_nr/cluster is lost, empty list will be
        returned.
        :return: list with the changed cluster count
        """
        changed_subspace = self.input_n_clusters.copy()
        for x in self.n_clusters:
            if x in changed_subspace:
                changed_subspace.remove(x)
        return changed_subspace

    def calculate_mdl_costs(self, X):
        """
        Calculate the Mdl Costs of this NrKmeans result.
        :param X: input data
        :return: total_costs, global_costs, all_subspace_costs
        """
        if self.labels_ is None:
            raise Exception("The NrKmeans algorithm has not run yet. Use the fit() function first.")
        return _mdl_costs(X, self)

    def _create_full_rotation_matrix(self, dimensionality, m, V_C):
        """
        Create full rotation matrix out of the found eigenvectors. Set diagonal to 1 and overwrite columns and rows with
        indices in P_combined (consider the oder) with the values from V_C. All other values should be 0.
        :param dimensionality: dimensionality of the full rotation matrix
        :param P_combined: combined projections of the subspaces
        :param V_C: the calculated eigenvectors
        :return: the new full rotation matrix
        """
        V_F = np.identity(dimensionality)
        V_F[np.ix_(range(m), range(m))] = V_C
        return V_F


# die Anzahl an dim, jedes einzelnen Clusterspaces, kann unterschiedlich sein
def myOrth1(X, all_num_clusters, random_state):
    """
    Execute the myOrth1 algorithm. The algorithm will search for non-redundant clustering views of the same data.
    :param X: input data
    :param all_num_clusters: list containing number of clusters for each subspace_nr
    :return:  pred_labels, total_costs_list, cluster_m, res_V_proj
    """
    sum_along_dimension = X.sum(axis=0)
    n_points = X.shape[0]
    data_dimensionality = X.shape[1]
    m_cnt = X.shape[1]
    cluster_m = []
    clusterlength = len(all_num_clusters)
    res_V = np.array([None for k in all_num_clusters])
    list_cluster_centers = [None for k in all_num_clusters]
    res_scatter_matrices = np.array([None for k in all_num_clusters])

    mean_values = sum_along_dimension / n_points
    listmeans = mean_values.tolist()
    listmeans = listmeans * n_points
    listmeans_array = np.asarray(listmeans)

    E = listmeans_array.reshape(n_points, data_dimensionality)
    data = X - E

    n_subspaces = len(all_num_clusters)
    pred_labels = np.zeros([n_points, n_subspaces])

    for s in range(n_subspaces):
        numberclusters_in_current_subspace = all_num_clusters[s]
        centers = np.array([None]*numberclusters_in_current_subspace)

        pca = PCA(random_state=random_state)
        score = pca.fit_transform(data.astype(np.float64)).astype(np.float64)
        latent = pca.explained_variance_.astype(np.float64)
        dim = (np.cumsum(latent) / np.sum(latent)).astype(np.float64)

        for i in range(len(dim)):
            if dim[i] >= 0.9:
                break

        # most important columns
        V_projected = score[:, :i+1]
        noisespace_columns = score[:, i+1:]

        # V anfügen
        V = pca.components_

        # Wenn Clusterlänge eins entspricht -> noise space
        if clusterlength > 1:
            if s == 0:
                cluster_m.append(V_projected.shape[1])
                m_cnt -= V_projected.shape[1]
            else:
                if (m_cnt - V_projected.shape[1]) > 0:
                    m_cnt -= V_projected.shape[1]
                    cluster_m.append(m_cnt)
                else:
                    cluster_m.append(m_cnt)
        else:
            cluster_m.append(data_dimensionality)

        print("V_proj: ", V_projected.shape[1])
        #print(f"V_shape: {V.shape}, Score_shape: {score.shape}, data: {data.shape}")

        kmeans = KMeans(n_clusters=numberclusters_in_current_subspace, max_iter=300, n_init='auto', random_state=random_state)

        #pred_labels = kmeans.fit_predict(np.asarray(V_projected).reshape(n_points, i + 1))
        pred_labels[:, s] = kmeans.fit_predict(np.asarray(V_projected).reshape(n_points, i + 1))

        # list containing prediction labels
        #centroids = kmeans.fit(np.asarray(V_projected).reshape(n_points, i + 1)).cluster_centers_
        # .reshape(n_points, data_dimensionality))
        centroids = kmeans.fit(np.asarray(V)).cluster_centers_

        mu = np.zeros((len(centroids), data_dimensionality)).astype(np.float64)

        # k viele Cluster in Clusterspace und ein Noisespace
        #n_clusters = [numberclusters_in_current_subspace, 1]
        #scatter_matrices = [None] * 2

        centers, scatter_matrices = _update_centers_and_scatter_matrices(data, numberclusters_in_current_subspace, pred_labels[:, s])

        # Speichere alle Zentren
        if numberclusters_in_current_subspace != 1:
            select_centers = np.zeros((numberclusters_in_current_subspace, data_dimensionality))
            for i in range(numberclusters_in_current_subspace):
                select_centers[i] = centers[i, :]
            list_cluster_centers[s] = select_centers
        else:
            list_cluster_centers[s] = centers
        res_scatter_matrices[s] = scatter_matrices

        # Daten transformieren
        for i1 in range(len(centroids)):
            tt = np.argwhere(pred_labels == i1)
            xasarray = np.squeeze(np.asarray(data)).astype(np.float64)
            tthelp = tt[0][0].astype(int)

            if len(tt) == 1:
                mu[i1, :] = xasarray[tthelp, :]
            else:
                test1 = []
                for i in range(len(tt)):
                    test1.append(xasarray[tt[i]])

                test2 = np.array(test1)
                d2 = test2.sum(axis=0)
                new_mean_values = d2 / len(test2)
                mu[i1] = new_mean_values[0]

        for i in range(n_points):
            b = mu[int(pred_labels[i, s])]
            newa = mu[int(pred_labels[i, s])].conj()
            hilfsmatrix_1 = np.divide(np.outer(np.transpose(newa), mu[int(pred_labels[i, s])]), np.dot(np.transpose(b), b))
            hilfsmatrix_2 = np.identity(mu.shape[1]) - hilfsmatrix_1
            data = np.asarray(X).reshape(n_points, data_dimensionality)
            data[i, :] = np.dot(X[i, :], hilfsmatrix_2)

    print(f"m_clusters: {cluster_m}, cluster_type: {type(list_cluster_centers)}")
    return pred_labels, list_cluster_centers, V, cluster_m, all_num_clusters,  res_scatter_matrices


def _check_number_of_points(labels_true, labels_pred):
    """
    Check if the predicted labels equals the number of points of the original labels
    :param labels_true: orginial labels
    :param labels_pred: predicted labels
    """
    if labels_pred.shape[0] != labels_true.shape[0]:
        raise Exception(
            "Number of objects of the prediction and ground truth are not equal.\nNumber of prediction objects: " + str(
                labels_pred.shape[0]) + "\nNumber of ground truth objects: " + str(labels_true.shape[0]))


def _remove_empty_cluster(n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace, debug):
    """
    Check if after label assignment and center update a cluster got lost. Empty clusters will be
    removed for the following rotation und iterations. Therefore all necessary lists will be updated.
    :param n_clusters_subspace: number of clusters of the subspace_nr
    :param centers_subspace: cluster centers of the subspace_nr
    :param scatter_matrices_subspace: scatter matrices of the subspace_nr
    :param labels_subspace: cluster assignments of the subspace_nr
    :return: n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace (updated)
    """
    # Check if any cluster is lost
    if np.any(np.isnan(centers_subspace)):
        # Get ids of lost clusters
        empty_clusters = np.where(np.any(np.isnan(centers_subspace), axis=1))[0]
        if debug:
            print(
                "[NrKmeans] ATTENTION: Clusters were lost! Number of lost clusters: " + str(
                    len(empty_clusters)) + " out of " + str(
                    len(centers_subspace)))
        # Update necessary lists
        n_clusters_subspace -= len(empty_clusters)
        for cluster_id in reversed(empty_clusters):
            centers_subspace = np.delete(centers_subspace, cluster_id, axis=0)
            scatter_matrices_subspace = np.delete(scatter_matrices_subspace, cluster_id, axis=0)
            labels_subspace[labels_subspace > cluster_id] -= 1
    return n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace


def _get_precision(X):
    precision_list = []
    for i in range(X.shape[1]):
        dist = cdist(X[:, i].reshape((-1, 1)), X[:, i].reshape((-1, 1)))
        dist_gt_0 = dist[dist > 0]
        if dist_gt_0.size != 0:
            precision_list.append(np.min(dist_gt_0))
    return np.mean(precision_list)
