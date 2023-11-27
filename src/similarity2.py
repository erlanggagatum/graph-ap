import networkx as nx
import numpy as np
import random
import math


# calculate euclidean distance
def calculate_euclidean_matrix(G):
    
    # n = len(G.nodes)
    # W = np.zeros((n, n))
    # node_features = np.array([G.nodes[i]['node_features'] for i in G.nodes])

    # for i in range(n):
    #     distances = np.linalg.norm(node_features - node_features[i], axis=1)
    #     W[i, :] = distances

    # return np.round(W, 1)

    # OLD
    W = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            val = round(np.linalg.norm(G.nodes[i]['node_features'] - G.nodes[j]['node_features']), 1)
            W[i][j] = val
            W[j][i] = val
    return W

def calculate_transition_probability_matrix(G):
    W = calculate_euclidean_matrix(G)  # Assuming this function is defined elsewhere
    
    # Initialize the probability matrix
    n = len(G.nodes)
    P = np.zeros((n, n))

    for i in G.nodes:
        neighbors = list(G.neighbors(i))
        denum = sum(W[i, neighbor] for neighbor in neighbors)
        if denum == 0:
            P[i, neighbors] = 1 / len(neighbors)
        else:
            P[i, neighbors] = [W[i, neighbor] / denum for neighbor in neighbors]

    return np.round(P, 2)

# def calculate_hitpath(G, s, e, max_iter = 100, transition_probability = None):
#     max_steps = G.number_of_nodes() // 2
#     sigma = 4 * G.number_of_nodes()  
#     hitting_distance = []
    
#     random.seed(42)  # Seed once, outside the loop
#     for _ in range(max_iter):
#         route = [s] 
#         for step_counter in range(max_steps):
#             if route[-1] == e:  # Check if last node is the end node
#                 break
#             neighbors = list(G.neighbors(route[-1]))
#             if neighbors:
#                 prob_neighbors = [transition_probability[route[-1]][neighbor] for neighbor in neighbors]
#                 chosen_node = random.choices(neighbors, weights=prob_neighbors, k=1)[0]
#                 route.append(chosen_node)

#         hitdist = step_counter if route[-1] == e else sigma
#         hitting_distance.append(hitdist)

#     hitpath = sum(hitting_distance) / max_iter  # Calculate average over iterations
#     return hitpath

def calculate_hitpath(G, s, e, max_steps = 10, max_iter = 100, transition_probability = None, W = None):
    if s == e:
        return 0
    
    max_steps = G.number_of_nodes()//2
    sigma = 4 * G.number_of_nodes()  
    # print(W[1][1])

    hitting_distance = []
    
    random.seed(42)  
    for _ in range(max_iter):
        # print(f'iter {_}')
        
        # save route
        route = [s] 
        step_counter = 0
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
        
        if e not in route:
            hitting_distance.append(sigma)
        else:
            hitdist = 0
            for i in range(len(route)-1):
                hitdist += W[route[i]][route[i+1]]
                # weight
            hitting_distance.append(hitdist)
            
    # after walk, calculate hitting distance
    hitpath = sum(hitting_distance)/(max_iter)
    
    return hitpath

def calculate_hitpath_old(G, s, e, max_steps = 10, max_iter = 100, transition_probability = None):
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
    
    random.seed(42)  
    for _ in range(max_iter):
        # print(f'iter {_}')
        
        # save route
        route = [s] 
        step_counter = 0
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
    # hitpath = sum(hitting_distance)/(max_steps)
    hitpath = sum(hitting_distance)/(max_steps)
    
    return hitpath

def calculate_hitpath_matrix(G, transition_probability = None):
    W = calculate_euclidean_matrix(G)
    H = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            H[i][j] = calculate_hitpath(G, i, j, max_iter=100, transition_probability=transition_probability, W=W)
            
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
    # W = calculate_euclidean_matrix(G)
    H = calculate_hitpath_matrix(G, transition_probability=transition_probability)
    # print('somting wong')
    precalc_shortest_path_length = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            precalc_shortest_path_length[i][j] = nx.shortest_path_length(G, i, j)

    
    S = np.zeros((len(G.nodes), len(G.nodes)))
    for i in G.nodes:
        for j in G.nodes:
            S[i][j] = calculate_similarity(G, H, i, j, gamma=gamma, 
                                           precalc_shortest_path_length=precalc_shortest_path_length)
            
    
    # print('somting wong')
    return S