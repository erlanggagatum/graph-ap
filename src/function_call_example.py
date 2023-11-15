import torch
import torch_geometric
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.datasets import TUDataset
from preprocessing import data_transformation
from similarity import calculate_similarity_matrix

from model import GCN

# PREPROCESSING: transform dataset into networkx data
dataset = TUDataset("/", name='MUTAG')
data = 0
print('dataset', dataset[1])

# print("DATASET: ", dataset)
# print(dataset[0])

# G = data_transformation(dataset[0].edge_index, dataset[0].x)

# # SIMILARITY MEASURMENT: using hitpath
# S = calculate_similarity_matrix(G)

# print(S)

# Calculate embedding
model = GCN(dataset, 64)
emb = model(dataset[data].x, dataset[data].edge_index)
print("embedding", emb)
emb = emb.detach()
G = data_transformation(dataset[data].edge_index, emb)
S = calculate_similarity_matrix(G)

# AP Clustering
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation(affinity='precomputed', random_state=123).fit(S)

    
pos = nx.spring_layout(G, seed=212)

print('cluster center: ', clustering.cluster_centers_indices_)

labels = {} 
for i, l in enumerate(clustering.labels_):
    labels[i] = str(i)+" - "+str(l)
    if i in clustering.cluster_centers_indices_:
        labels[i] = labels[i] +" ++"
    
nx.draw(G, pos=pos)
nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="black")
plt.show()
