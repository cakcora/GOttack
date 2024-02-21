
import numpy as np
import scipy.sparse as sp
import torch
import torch.sparse as ts
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from collections import deque
import networkx as nx


def get_linearized_weight(surrogate):
    W = surrogate.gc1.weight @ surrogate.gc2.weight
    return W.detach().cpu().numpy()

def normalize_adj(mx):
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def edg_graph(nodes, thisdict):
    thisdict=thisdict
    nodes=nodes
    compared_orbit_names = []
    for orbit_pair_key in thisdict.keys():
        compared_orbit_key = f"{orbit_pair_key}"
        if len(thisdict[compared_orbit_key]) > 0:
            compared_orbit_names.append(orbit_pair_key)
    G=nx.Graph()
    for i in range(len(nodes)):
        G.add_node(i,  ftlst= thisdict[compared_orbit_names[i]])
        con=nodes[i]['connections']
        d = {i: con}
        new_list = [(k, v) for k in d for v in d[k]]
        G.add_edges_from(new_list)
    return G

###########  function ##################
def f_exploration(s, n):
  if n==0:
    f_value =1000000
  else:
    f_value=(s+3/n)
  return f_value

def compute_logits(work_modified_adj, modified_features, surrogate, target_node):
    adj_norm = normalize_adj(work_modified_adj)
    W = get_linearized_weight(surrogate)
    logits = (adj_norm @ adj_norm @ modified_features @ W)[target_node]
    return logits


def find_all_2hop_neighbors1(graph):
    all_2hop_neighbors = {}

    for node in graph:
        visited = set()
        neighbors = set()
        queue = deque([(node, 0)])  # Use a queue for BFS traversal with level information

        while queue:
            current_node, level = queue.popleft()
            visited.add(current_node)

            if level == 2:  # Reached the 2-hop level
                neighbors.add(current_node)

            elif level < 2:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, level + 1))

        all_2hop_neighbors[node] = list(neighbors)

    return all_2hop_neighbors

def select_lists_with_element1(list_of_lists, element):
    selected_lists = [sublist for sublist in list_of_lists if element in sublist]
    return selected_lists