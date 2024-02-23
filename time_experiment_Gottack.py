import pandas as pd
import time
import statistics as stats
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from tqdm import tqdm
from deeprobust.graph.data import Dataset
import warnings
warnings.filterwarnings("ignore")
from OrbitAttack import OrbitAttack
from orbit_table_generator import OrbitTableGenerator
import random

"""
Author: Ngo Bao, Zulfikar
Credits:    The OrbitAttack is implemented with reference to Nettack's source code,
            a method proposed in the paper: 'Adversarial Attacks on Neural Networks for Graph Data'
            by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann
"""

random.seed(10)
dataset_name = 'cora'
df_2d = OrbitTableGenerator(dataset_name).generate_orbit_table()
device = "cpu"


def test_acc_GCN(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()



def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


######################### Loading dataset  #########################
data = Dataset(root='dataset', name=dataset_name)
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

idx_unlabeled = np.union1d(idx_val, idx_test)

######################### Setup GCN Surrogate model  #########################

surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)



miss= np.zeros((6, 10))
node_list = select_nodes()

num = len(node_list)

nofnode_modification_lst = [1]

time_result = []
for i in range(5):

# Start the timer
    start_time = time.time()

    for _ in range(len(nofnode_modification_lst)):
        nofnode_modification = nofnode_modification_lst[_]
        cnt1=0
        edg_pub_rate = nofnode_modification  # 0.
        degrees = adj.sum(0).A1
        num = len(node_list)
        bst_edge={}

        for target_node in tqdm(node_list):
            n_perturbations = edg_pub_rate #2 #math.ceil(degrees[target_node] * edg_pub_rate)  # int(degrees[target_node] * edg_pub_rate)

            model = OrbitAttack(surrogate,df_2d, nnodes=adj.shape[0], device=device)
            model = model.to(device)
            model.attack(features, adj, labels, target_node, 1, verbose=False)
            modified_adj = model.modified_adj
            acc = test_acc_GCN(modified_adj, features, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            bst_edge[target_node] = [model.best_edge_list, acc]
            if acc == 0:
                cnt1 += 1

        miss[1][_] = cnt1 / num
        print( miss[1][_])

    ######################################## ++++++++++    END   ++++++++++ For Testing the model ##########################

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    time_result.append(elapsed_time)
print("-------------------------------------Result---------------------------------")
print(time_result)
print("Time experiment (5 times running):" + str(round(sum(time_result)/len(time_result),2)) + "STD: " + str(round(stats.stdev(time_result),2)))


