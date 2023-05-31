from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#tf.debugging.set_log_device_placement(True)

from pygsp import graphs, filters, plotting
from _datetime import datetime
import numpy as np
import pickle


# 1-hope [3,3],  2-hope [5,5] and 3-hope [7,7]
IMAGE_SHAPE = [128,128]   # H x W

class Vertex:
    def __init__(self, n):
        self.name = n
        self.neighbors = list()

        self.distance = 9999
        self.color = 'black'

    def add_neighbor(self, v):
        if v not in self.neighbors:
            self.neighbors.append(v)
            self.neighbors.sort()


class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
            self.vertices[vertex.name] = vertex
            return True
        else:
            return False

    def add_edge(self, u, v):
        if u in self.vertices and v in self.vertices:
            for key, value in self.vertices.items():
                if key == u:
                    value.add_neighbor(v)
                if key == v:
                    value.add_neighbor(u)
            return True
        else:
            return False

    def print_graph(self,hop):
        for key in sorted(list(self.vertices.keys())):
            #print(str(key) + str(self.vertices[key].neighbors) + "  " + str(self.vertices[key].distance))
            if self.vertices[key].distance == hop:
                print("{}-hop neighbor: {}".format(hop,(str(key) + str(self.vertices[key].neighbors) + "  " + str(self.vertices[key].distance))))


    def bfs(self, vert, hop):
        q = list()
        n = list()
        bbox = list()
        vert.distance = 0
        vert.color = 'red'
        n.append(vert.name)
        for v in vert.neighbors:
            self.vertices[v].distance = vert.distance + 1
            # Add this for all hope belongs to N
            n.append(self.vertices[v].name)
            q.append(v)

        while len(q) > 0:
            u = q.pop(0)
            node_u = self.vertices[u]
            node_u.color = 'red'

            for v in node_u.neighbors:
                node_v = self.vertices[v]
                if node_v.color == 'black':
                    # This comment is for fixing 2-hop , 3-hop problem
                    #q.append(v)
                    if node_v.distance > node_u.distance + 1:
                        node_v.distance = node_u.distance + 1
                        if node_v.distance == hop:
                            n.append(node_v.name)
                            bbox.append(node_v.name)
                        else:
                            n.append(node_v.name)
                            q.append(v)
        return n, bbox

def initilize(g,edges):
    '''
    g = Graph()
    for i in range(0, (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])):
        g.add_vertex(Vertex(str(i)))
    for edge in edges:
        g.add_edge(str(edge[0]), str(edge[1]))
    '''
    with open("g.pkl", "rb") as fp:
        g = pickle.load(fp)

    return g

def run(hop, g, edges,BFS_pred):
    mask_bbox = []
    g = initilize(g,edges)
    a = g.vertices.get(str(BFS_pred))
    n, bbox = g.bfs(a,hop)
    MB = (n, bbox)
    mask_bbox.append(MB)
    return mask_bbox
#--------------------------
# MAIN
#--------------------------
#if __name__ == '__main__':
def main(DELTA_value, nhop):
    start = datetime.now()
    #print("Start Time",start)

    '''
    G = graphs.Grid2d(IMAGE_SHAPE[0],IMAGE_SHAPE[1])                                                           # make sure x[0] having dimension nVertices x nChannel
    A = G.A

    mtr = A.tocoo()

    edg = np.stack((mtr.row, mtr.col), axis=-1)

    with open("edg.pkl", "wb") as fp:
        pickle.dump(edg, fp)

    g = Graph()

    # g.add_vertex(Vertex('B'))
    for i in range(0, (IMAGE_SHAPE[0]*IMAGE_SHAPE[1])):
        g.add_vertex(Vertex(str(i)))

    #edges = ['AB', 'AE', 'BF', 'CG', 'DE', 'DH', 'EH', 'FG', 'FI', 'FJ', 'GJ', 'HI']
    edges = edg

    # load graph edges g and comment below to save time
    # with open("g.pkl", "rb") as fp:
    #     g = pickle.load(fp)

    for edge in edges:
        g.add_edge(str(edge[0]), str(edge[1]))

    with open("g.pkl", "wb") as fp:
        pickle.dump(g, fp)
    '''

    with open("g.pkl", "rb") as fp:
        g = pickle.load(fp)

    with open("edg.pkl", "rb") as fp:
        edg = pickle.load(fp)

    hop_MB = []
    for hop in range(2,nhop):
        MB = {}
        MB[str(hop)] = {}
        #print("Hop number:{}".format(hop))
        mask_bbox = run(hop, g, edg, DELTA_value)
        MB = {hop:mask_bbox}   #[str(hop)][str(hop)].append({str(hop):mask_bbox})
        hop_MB.append(MB)
    return hop_MB