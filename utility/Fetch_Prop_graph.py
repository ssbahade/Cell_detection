import os,sys,inspect
import os
import joblib
import tensorflow as tf
import numpy as np
import h5py
import scipy.sparse.linalg as la
import scipy.sparse as sp
import scipy
import time
import pickle
from sklearn.utils import shuffle

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

import scipy.io as sio

from utility import graph
from utility import coarsening
from utility import utils

# Graphs.
number_edges = 4
metric ='euclidean'
normalized_laplacian = True
coarsening_levels = 0
len_img = 128

np.random.seed(0)

# Useful functions

def grid_graph(m):
    z = graph.grid(m)  # normalized nodes coordinates
    dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
    # dist contains the distance of the 8 nearest neighbors for each node indicated in z sorted in ascending order
    # idx contains the indexes of the 8 nearest for each node sorted in ascending order by distance

    A = graph.adjacency(dist,idx)  # graph.adjacency() builds a sparse matrix out of the identified edges computing similarities as: A_{ij} = e^(-dist_{ij}^2/sigma^2)

    return A, z


# A friend helped me a lot with this function, so you will find something similar in other assignment
def coarsen_mnist(A, levels, nodes_coordinates):
    graphs, parents = coarsening.metis(A,
                                       levels)  # Coarsen a graph multiple times using Graclus variation of the METIS algorithm.
    # Basically, we randomly sort the nodes, we iterate on them and we decided to group each node
    # with the neighbor having highest w_ij * 1/(\sum_k w_ik) + w_ij * 1/(\sum_k w_kj)
    # i.e. highest sum of probabilities to randomly walk from i to j and from j to i.
    # We thus favour strong connections (i.e. the ones with high weight wrt all the others for both nodes)
    # in the choice of the neighbor of each node.

    # Construction is done a priori, so we have one graph for all the samples!

    # graphs = list of spare adjacency matrices (it contains in position
    #          0 the original graph)
    # parents = list of numpy arrays (every array in position i contains
    #           the mapping from graph i to graph i+1, i.e. the idx of
    #           node i in the coarsed graph -> that is, the idx of its cluster)
    perms = coarsening.compute_perm(parents)  # Return a list of indices to reorder the adjacency and data matrices so
    # that two consecutive nodes correspond to neighbors that should be collapsed
    # to produce the coarsed version of the graph.
    # Fake nodes are appended for each node which is not grouped with anybody else

    coordinates = np.copy(nodes_coordinates)
    u_shape, u_rows, u_cols, u_val = [], [], [], []

    for i, A in enumerate(graphs):
        M, M = A.shape

        # We remove self-connections created by metis.
        A = A.tocoo()
        A.setdiag(0)

        if i < levels:  # if we have to pool the graph
            A = coarsening.perm_adjacency(A, perms[i])  # matrix A is here extended with the fakes nodes
            # in order to do an efficient pooling operation
            # in tensorflow as it was a 1D pooling

        A = A.tocsr()
        A.eliminate_zeros()

        Mnew, Mnew = A.shape
        u_shape.append(Mnew)

        if i == 0:
            fake_nodes = Mnew - M
            coordinates = np.concatenate([coordinates, np.ones([fake_nodes, 2]) * np.inf], 0)
            if i < levels:
                coordinates = coordinates[perms[0]]

        start_node, end_node = A.nonzero()
        u_rows.append(start_node)
        u_cols.append(end_node)

        distance = coordinates[start_node] - coordinates[end_node]
        u_val.append(distance)

        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added), |E| = {3} edges'.format(i, Mnew, Mnew - M, A.nnz // 2))

        # update coordinates for next coarser graph
        new_coordinates = []
        for k in range(A.shape[0] // 2):
            idx_first_el = k * 2

            if not np.isfinite(coordinates[idx_first_el][0]):
                new_coordinates.append(coordinates[idx_first_el + 1])

            elif not np.isfinite(coordinates[idx_first_el + 1][0]):
                new_coordinates.append(coordinates[idx_first_el])

            else:
                new_coordinates.append(np.mean(coordinates[idx_first_el:idx_first_el + 2], axis=0))

        coordinates = np.asarray(new_coordinates)

    return u_shape, u_rows, u_cols, u_val


# Create u

n_rows_cols = len_img
A, nodes_coordinates = grid_graph(n_rows_cols)

u_shape, u_rows, u_cols, u_val = coarsen_mnist(A, coarsening_levels, nodes_coordinates)

u = []
if coarsening_levels != 0:
    for level in range(coarsening_levels):
        u.append([u_shape[level], u_rows[level], u_cols[level], u_val[level]])
else:
    u.append([u_shape[0], u_rows[0], u_cols[0], u_val[0]])


def Fetch_graph(proposals):
    ################  find proposal graph and pseudocordinates ############
    pseudo_node_num = u[0][1]

    N_IDX = np.array([],dtype=np.int32)
    for i in proposals:
        node_idx = np.where(pseudo_node_num == i)
        N_IDX = np.append(N_IDX,node_idx[0])

    propo_u_1 = u[0][1][N_IDX]
    propo_u_2 = u[0][2][N_IDX]
    propo_u_3 = u[0][3][N_IDX]

    No_edge_idx = np.where(np.isin(propo_u_2, propo_u_1)==False)

    propo_graph_u_1 = np.delete(propo_u_1, No_edge_idx[0])
    propo_graph_u_2 = np.delete(propo_u_2, No_edge_idx[0])
    propo_graph_u_3 = np.delete(propo_u_3, No_edge_idx[0], axis=0)
    propo_graph = [propo_graph_u_1, propo_graph_u_2, propo_graph_u_3]

    return propo_graph

#print(x_train.shape)

## tensor verson of from object_detection.Fetch_Prop_graph import Fetch_graph

def Fetch_graph_ver2(proposals):
    ################  find proposal graph and pseudocordinates ############
    pseudo_node_num = u[0][1]

    N_IDX = np.array([],dtype=np.int32)
    for i in proposals:
        node_idx = np.where(pseudo_node_num == i)
        N_IDX = np.append(N_IDX,node_idx[0])

    propo_u_1 = u[0][1][N_IDX]
    propo_u_2 = u[0][2][N_IDX]
    propo_u_3 = u[0][3][N_IDX]

    No_edge_idx = np.where(np.isin(propo_u_2, propo_u_1)==False)

    propo_graph_u_1 = np.delete(propo_u_1, No_edge_idx[0])
    propo_graph_u_2 = np.delete(propo_u_2, No_edge_idx[0])
    propo_graph_u_3 = np.delete(propo_u_3, No_edge_idx[0], axis=0)
    propo_graph = [propo_graph_u_1, propo_graph_u_2, propo_graph_u_3]

    return propo_graph
