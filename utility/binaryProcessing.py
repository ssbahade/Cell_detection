import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from scipy.spatial import Delaunay
from sklearn.model_selection import train_test_split
#from numba import vectorize, cuda, float32
from scipy.sparse import csr_matrix
from _datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os

from utility.Fetch_Prop_graph import Fetch_graph

import numpy as np
import random

def nclass_to_binclass(DATA):
    photos = np.asarray(DATA[0])
    labels = np.array(DATA[1])
    graphs = np.array(DATA[2])

    # Data preprocessing to arrange three class proposals into two class i.e object/ Non-object

    # combine object classes
    print("number of unique elements LABELS : {}".format(np.unique(labels, return_counts=True)))
    b = np.unique(labels, return_counts=True)
    print("shape",np.shape(b))

    #comment below to balance 3 class data to binalry class i.r object/Non-object
    ## balance the classese with the smallest number of sample for three classes
    
    class_0_indices = (np.where(labels == 0))[0]     # BG class index location
    class_2_indices = (np.where(labels == 2))[0]     # blue class index location
    class_1_indices = (np.where(labels == 1))[0]     # Red class index location
    # #
    shuffle_class_0_indices = np.random.permutation(class_0_indices)         # shuffle class 0 indices for better learning
    shuffle_class_1_indices = np.random.permutation(class_1_indices)         # shuffle class 1 indices for better learning

    #################### delete for two class #########
    # cls_idx= len(shuffle_class_1_indices)+len(class_2_indices)
    # CLASS_IDX = np.concatenate((shuffle_class_0_indices[:cls_idx], shuffle_class_1_indices, class_2_indices))
    CLASS_IDX = np.concatenate((shuffle_class_0_indices, shuffle_class_1_indices, class_2_indices))
    random.shuffle(CLASS_IDX)                          # shuffle all the indices
    #
    photos = photos[CLASS_IDX]
    labels = labels[CLASS_IDX]
    graphs = graphs[CLASS_IDX]

    Adjacency = []
    for gr in graphs:
        source = gr[0]
        dest = gr[1]

        dest_new = np.copy(dest)

        # replace source nodes with unique nodes starting from 0
        src_unq_idx_cnt = np.unique(source, return_index=True, return_inverse= True, return_counts=True)
        g = [a for a in range(len(src_unq_idx_cnt[0]))]
        g_idx = src_unq_idx_cnt[2]
        g1 = np.array(g)[g_idx.astype(int)]

        dst_unq_idx_cnt = np.unique(dest, return_index=True, return_inverse= True, return_counts=True)
        dst_unq = dst_unq_idx_cnt[0]
        for du in dst_unq:
            idx = np.where(dest==du)

            idx_s = np.where(source == du)
            real_s = g1[idx_s[0]]

            # put actual value to the destination
            dest_new[idx[0]]=real_s

        new_gr = [g1,dest_new]
        new_gr = np.array(new_gr)

        # create a binary adjaceny matrix using csr_matrix
        N = len(src_unq_idx_cnt[0])   #num of nodes
        value = np.ones(len(new_gr[0]))

        Adj = csr_matrix((value, (new_gr[0], new_gr[1])), shape=(N, N), dtype=np.float32).toarray()

        Adjacency.append(Adj)
    Adjacency = np.array(Adjacency)


    A_train, A_test, x_train, x_test, y_train, y_test = train_test_split(Adjacency, photos, labels, test_size=0.1)

    return A_train, A_test, x_train, x_test, y_train, y_test
