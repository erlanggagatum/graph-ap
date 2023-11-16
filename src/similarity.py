import networkx as nx
import numpy as np
import random
import math


# calculate euclidean distance
def calculate_euclidean_matrix(G):
    W = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            val = round(np.linalg.norm(G.nodes[i]['node_features'] - G.nodes[j]['node_features']), 1)
            W[i][j] = val
            W[j][i] = val
                
    return W

def calculate_transition_probability_matrix(G):
    W = calculate_euclidean_matrix(G)
    
    P = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        neighbors = list(G.neighbors(i))
        denum = 0
        for neighbor in neighbors:
            denum += W[i][neighbor]
        for neighbor in neighbors:
            if denum == 0:
                P[i][neighbor] = round(1/len(neighbors), 2)
                continue
            P[i][neighbor] = round(W[i][neighbor]/denum, 2)
    return P

def calculate_hitpath(G, s, e, max_steps = 10, max_iter = 100, transition_probability = None):
    max_steps = G.number_of_nodes()//2
    
    # 
    # transition_probability = calculate_transition_probability_matrix(G)
    # 
    # 
    # 
    # 
    
    
    # print(transition_probability)
    sigma = 4 * G.number_of_nodes()  
    
    hitting_distance = []
    
    for _ in range(max_iter):
        # print(f'iter {_}')
        
        # save route
        route = [s] 
        step_counter = 0
        
        # print(s)
        hitpath = 0
        
        # do random walk bounded by max steps
        while (step_counter < max_steps):
            prob_neighbors = []
            if route[-1] in G:
                # see neighbors
                neighbors = list(G.neighbors(route[-1]))
                if neighbors:
                    if len(neighbors) > 1:
                        # choose step based on transition probability
                        # print('calculate prob', neighbors)
                        for neighbor in neighbors:
                            prob_neighbors.append(transition_probability[route[-1]][neighbor])
                        choosen_node = neighbors[random.choices(range(len(prob_neighbors)), weights=prob_neighbors, k=1)[0]]
                        route.append(choosen_node)
                    elif len(neighbors) == 1:
                        # can walk directly
                        # print('walk')
                        route.append(neighbors[0])
                        # print('walk, ', route)
                
            step_counter+=1
            if (e in route):
                break
        if step_counter == max_steps:
            hitting_distance.append(sigma)
        else:
            hitdist = 0
            for i in range(len(route)-1):
                hitdist += 1
            hitting_distance.append(hitdist)
        # after walk, calculate hitting distance
        hitpath = sum(hitting_distance)/(max_steps)
        # print(_, " --> Route: ", route, "max exceeded" if step_counter == max_steps else "reached, H=", hitting_distance,hitpath)
    hitpath = sum(hitting_distance)/(max_steps)
    
    return hitpath

def calculate_hitpath_matrix(G, transition_probability = None):
    H = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            H[i][j] = calculate_hitpath(G, i, j, max_iter=100, transition_probability=transition_probability)
    return H

def testt():
    print('ini bisa blooookkkk')

def calculate_similarity(G, H, s, e, gamma = 0.8, precalc_shortest_path_length = None):
    '''
    G = graph,
    H = Hitpath matrix of graph,
    s = start node,
    e = end node,
    gamma = (optional - default 0.8),
    '''
    
    similarity = 0
    # shortest_path_length = nx.shortest_path_length(G, s, e)
    shortest_path_length = precalc_shortest_path_length[s][e]

    # print(similarity)
    similarity = (gamma * math.exp(-shortest_path_length)) + ((1-gamma) * ((H[s][e] - H[e][s])**2))
    
    return similarity


# precalc_shortest_path_length = None

def calculate_similarity_matrix(G, gamma = 0.8):
    transition_probability = calculate_transition_probability_matrix(G)
    H = calculate_hitpath_matrix(G, transition_probability=transition_probability)
    
    # print('somting wong')
    precalc_shortest_path_length = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            precalc_shortest_path_length[i][j] = nx.shortest_path_length(G, i, j)

    
    S = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            S[i][j] = calculate_similarity(G, H, i, j, gamma=0.8, 
                                           precalc_shortest_path_length=precalc_shortest_path_length)
            
    
    # print('somting wong')
    return S