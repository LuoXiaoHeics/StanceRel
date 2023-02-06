# from random import random
from tqdm import tqdm
import numpy as np
import os.path, pickle
from random import sample
import  os, numpy as np
import random
import pandas as pd
from datetime import datetime, timezone
from utils_graph import conceptnet_graph

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_nodes(data):
    data = data.sort_values(by=['datetime'])
    parent = set(data['author_parent'])
    child = set(data['author_child'])
    nodes = parent.union(child)
    return nodes

if __name__ == '__main__':
    data = pd.read_csv("labeled_data.csv", index_col = False)
    all_nodes = get_nodes(data)
    unique_nodes_mapping = {}

    for item in all_nodes:
        unique_nodes_mapping[item] = len(unique_nodes_mapping)

    relation_matrix = {}
    meet_matrix = {}
    relation_map = {'friend':1,'enemy':2,'neutral':0,'interact':3}

    

    data = data.sort_values(by=['datetime'])
    print(len(unique_nodes_mapping))    
    print(data.shape)

    train = data.iloc[0:int(len(data)*0.8)]  

    parent = (train['author_parent'])
    child = (train['author_child'])
    relations = train['label']
    triplets = []
    for p in enumerate(zip(child,parent,relations)):
        p=p[1]
        if (unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]]) in relation_matrix:
            relation_matrix[(unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]])] += p[2]-1
        else : relation_matrix[(unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]])] =p[2]-1


    concept_net = {}
    for u in (unique_nodes_mapping.values()):
        concept_net[u] = []
    for key in relation_matrix.keys():
        if relation_matrix[key] > 0:
            if random.random()>0.3:
                triplets.append((key[0],1,key[1]))
                concept_net[key[0]].append([key[0],1,key[1]])
            else: 
                triplets.append((key[0],3,key[1]))
                concept_net[key[0]].append([key[0],3,key[1]])
        elif relation_matrix[key] < 0:
            if random.random()>0.3:
                triplets.append((key[0],2,key[1]))
                concept_net[key[0]].append([key[0],2,key[1]])
            else: 
                triplets.append((key[0],3,key[1]))
                concept_net[key[0]].append([key[0],3,key[1]])
        else:
            if random.random()>0.3:
                triplets.append((key[0],0,key[1]))
                concept_net[key[0]].append([key[0],0,key[1]])
            else: 
                triplets.append((key[0],3,key[1]))
                concept_net[key[0]].append([key[0],3,key[1]])

    test = data.iloc[int(len(data)*0.8):-1]
    parent = (test['author_parent'])
    child = (test['author_child'])
    relations = test['label']
    i = 0
    for p in enumerate(zip(child,parent,relations)):
        
        p=p[1]
        if not ((unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]]) in relation_matrix):
            triplets.append((unique_nodes_mapping[p[0]],3,unique_nodes_mapping[p[1]]))
            relation_matrix[(unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]])] = 10000
            concept_net[unique_nodes_mapping[p[0]]].append((unique_nodes_mapping[p[0]],3,unique_nodes_mapping[p[1]]))
            i+=1

    triplets = np.array(triplets)

    if not os.path.exists('utils'):
        os.mkdir('utils')

    pickle.dump(all_nodes, open('utils/all_n_nodes_bert.pkl', 'wb'))
    pickle.dump(relation_map, open('utils/relation_n_map_bert.pkl', 'wb'))
    pickle.dump(unique_nodes_mapping, open('utils/unique_nodes_n_mapping_bert.pkl', 'wb'))
    pickle.dump(concept_net, open('utils/concept_n_graphs_bert.pkl', 'wb'))
    np.ndarray.dump(triplets, open('utils/triplets_n_bert.np', 'wb'))        