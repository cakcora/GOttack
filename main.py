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
from deeprobust.graph.data import Dataset, Dpr2Pyg
from model.GIN import GIN
from model.GSAGE import GraphSAGE
from deeprobust.graph.targeted_attack import Nettack

"""
Author: Ngo Bao, Zulfikar
Credits:    The OrbitAttack is implemented with reference to Nettack's source code,
            a method proposed in the paper: 'Adversarial Attacks on Neural Networks for Graph Data'
            by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann
"""

random.seed(10)
dataset_name = 'cora'
df_orbit = OrbitTableGenerator(dataset_name).generate_orbit_table()
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

def test_acc_GIN(adj,features, data, target_node):
    ''' test on GIN '''
    pyg_data = Dpr2Pyg(data)
    gin = GIN(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gin = gin.to(device)
    perturbed_adj = adj.tocsr()
    pyg_data.update_edge_index(perturbed_adj)
    gin.fit(pyg_data, verbose=False)
    gin.eval()
    output = gin.predict()
    acc_test = (output.argmax(1)[target_node] == labels[target_node])

    return acc_test.item()

def test_GSAGE(adj,features, data, target_node):
    ''' test on GSAGE '''
    pyg_data = Dpr2Pyg(data)
    gsage = GraphSAGE(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gsage = gsage.to(device)
    perturbed_adj = adj.tocsr()           ###########   perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    gsage.fit(pyg_data, verbose=False)
    gsage.eval()
    output = gsage.predict()
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

def set_up_surrogate_model(features, adj, labels, idx_train, idx_val):
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0, with_relu=False,
                    with_bias=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate

def attack (attack_model, target_node_list,budget,features,adj,labels,test_model = 'GCN',verbose = False):
    miss = 0
    for target_node in tqdm(target_node_list):
        attack_model.attack(features, adj, labels, target_node, budget, verbose=verbose)
        modified_adj = attack_model.modified_adj
        if test_model == 'GCN':
            acc = test_acc_GCN(modified_adj, features,
                               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        if acc == 0:
            miss += 1
    return miss / len(target_node_list)

######################### Loading dataset  #########################
data = Dataset(root='dataset', name=dataset_name)
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

idx_unlabeled = np.union1d(idx_val, idx_test)

######################### Setup GCN Surrogate model  #########################



method = ['1518','Nettack']
budget_list = [1]
rowlist = []

for budget in budget_list:
    row = []
    target_node = select_nodes()

    #Orbit attack(1518)
    surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val)
    model = OrbitAttack(surrogate, df_orbit, nnodes=adj.shape[0], device=device)
    model = model.to(device)
    miss_percentage = attack(model, target_node,budget,features,adj,labels)
    row.append(miss_percentage)

    # Nettack
    surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val)
    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
    model = model.to(device)
    miss_percentage = attack(model, target_node, budget, features, adj, labels)
    row.append(miss_percentage)


    rowlist.append(row)

result = pd.DataFrame(rowlist,columns=method,index=budget_list)
result.to_csv('./results/result.csv')








