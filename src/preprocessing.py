import networkx as nx
import numpy as np

def data_transformation(edge_index, node_features):
    # initialize nx.Graph() object
    G = nx.Graph()
    for start, end in zip(*edge_index):
        G.add_edge(start.item(), end.item())
    
    # set node features
    feat_dic = {}
    # print(node_features)
    for index, feat in enumerate(node_features):
        # print(feat.numpy())
        feat_dic[index] = np.array(feat.numpy())
    nx.set_node_attributes(G, values=feat_dic, name="node_features")
    # print(G.edges)
    return G


# edge_index = [ 
#     [ 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14, 15, 16 ], 
#     [ 1, 5, 0, 2, 1, 3, 2, 4, 9, 3, 5, 6, 0, 4, 4, 7, 6, 8, 7, 9, 13, 3, 8, 10, 9, 11, 10, 12, 11, 13, 14, 8, 12, 12, 15, 16, 14, 14 ] 
# ]

# node_features = [ 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 1, 0, 0, 0, 0, 0, 0 ], 
#     [ 0, 1, 0, 0, 0, 0, 0 ], 
#     [ 0, 0, 1, 0, 0, 0, 0 ], 
#     [ 0, 0, 1, 0, 0, 0, 0 ] 
# ]

# G = data_transformation(edge_index, node_features)