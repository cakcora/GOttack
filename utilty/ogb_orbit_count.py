import math

import networkx as nx
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
from orbit_table_generator import OrbitTableGenerator, generate_orbit_tables_from_count
import orca
from torch_geometric.utils import to_networkx

# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch_geometric.data import DataLoader


dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
dataset.get_idx_split()
edge_index = dataset[0].edge_index

G = nx.Graph()

# Add edges to the graph
for i in range(edge_index.size(1)):
    src = edge_index[0, i].item()
    dst = edge_index[1, i].item()
    G.add_edge(src, dst)

print("Generating graph")
G = to_networkx(dataset[0], to_undirected=True)
print("Done generating graph")
print(len(list(G.nodes)))


orbit_counts = orca.orbit_counts("node", 5, G)
print("Done counting orbit")
orbit_df = generate_orbit_tables_from_count(orbit_counts,sorted(list(G.nodes)))
orbit_df.to_csv("arxiv_orbit_df.csv")









