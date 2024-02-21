import pandas as pd
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *

####### Read the Graphlet data from the file

with open('../dataset/orbit/dpr_cora.out', 'r') as file:
    lines = file.readlines()

# Process the lines to create a list of lists
data_list = [list(map(int, line.split())) for line in lines]

# Convert the list of lists to a NumPy array

graphlet_features = np.array(data_list)
mylist = []
for i in range(len(graphlet_features)):
    arr = graphlet_features[i]
    sorted_indices = np.argsort(arr)[::-1]

    if sorted_indices[0] < sorted_indices[1]:
        s1 = str(sorted_indices[0]) + str(sorted_indices[1])
    else:
        s1 = str(sorted_indices[1]) + str(sorted_indices[0])

    mylist.append([i, sorted_indices[0], sorted_indices[1], s1])

my_array = np.array(mylist)
df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
print(df_2d)

######################### Loding the dataset   #########################
data = Dataset(root='dataset', name='cora')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

######################### Setup Surrogate model  #########################
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

nclass=labels.max().item()+1