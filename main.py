import pandas as pd
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from tqdm import tqdm
import warnings

from OrbitAttack import OrbitAttack

warnings.filterwarnings("ignore")
from orbit_table_generator import OrbitTableGenerator
import random
from deeprobust.graph.data import Dataset, Dpr2Pyg
from model.GIN import GIN
from model.GSAGE import GraphSAGE
from deeprobust.graph.targeted_attack import Nettack

"""
Author: Anonymous Authors
Credits:    
            - The OrbitAttack is implemented with reference to Nettack's source code,
            a method proposed in the paper: 'Adversarial Attacks on Neural Networks for Graph Data'
            by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann
            - Gottack is built on top of DeepRobust - A PyTorch Library for Adversarial
            Attacks and Defenses developed by Yaxin Li, Wei Jin, Han Xu and Jiliang Tang
"""



def test_acc_GCN(adj, features, data,target_node):
    idx_train, idx_val = data.idx_train, data.idx_val
    labels = data.labels
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
    labels = data.labels

    ''' test on GIN '''
    # reset feature to 0------------------------Remove this line if you don't want to feed GIN with node features.
    # data.features = sp.csr_matrix(data.features.shape, dtype=int)
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

def test_acc_GSAGE(adj,features, data, target_node):
    labels= data.labels
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

def attack (data,attack_model, target_node_list,budget,features,adj,labels,test_model,verbose = False):
    miss = 0
    false_class_node = []
    for target_node in tqdm(target_node_list):
        attack_model.attack(features, adj, labels, target_node, budget, verbose=verbose)
        modified_adj = attack_model.modified_adj
        if test_model == 'GCN':
            acc = test_acc_GCN(modified_adj, features,data,
                               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        elif test_model == 'GIN':

            acc = test_acc_GIN(modified_adj, features,data,
                               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        elif test_model == 'GSAGE':
            acc = test_acc_GSAGE(modified_adj, features,data,
                               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        else:
            raise Exception("Test model is not supported")

        if target_node in false_class_node:
            miss += 1
        elif acc == 0:
            miss += 1
            false_class_node.append(target_node)
    return miss / len(target_node_list)

def attack_net(surrogate,data, target_node_list,budget,features,adj,labels,test_model,verbose = False):
    miss = 0
    false_class_node = []
    for target_node in tqdm(target_node_list):
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, budget, verbose=verbose)
        modified_adj = model.modified_adj
        if test_model == 'GCN':
            acc = test_acc_GCN(modified_adj, features, data,
                               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        elif test_model == 'GIN':

            acc = test_acc_GIN(modified_adj, features, data,
                               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        elif test_model == 'GSAGE':
            acc = test_acc_GSAGE(modified_adj, features, data,
                                 target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        else:
            raise Exception("Test model is not supported")
        if target_node in false_class_node:
            miss += 1
        elif acc == 0:
            miss += 1
            false_class_node.append(target_node)
    return miss / len(target_node_list)


if __name__ == '__main__':
    '''
       Config attack method and dataset
       method : list of adversarial techniques (1518 is Gottack)
       budget list: Perturbation budget list
       dataset_name : cora or citeseer or pubmed or polblogs or Blogcatalog 
       
    '''
    method = ['1518', 'Nettack']
    budget_list = [1,2,3,4,5]
    random.seed(102)
    dataset_name = 'cora'
    df_orbit = OrbitTableGenerator(dataset_name).generate_orbit_table()
    device= "cpu"

    print("INFO: Applying adversarial techniques {} on {} dataset with perturbation budget {} ".format(method, dataset_name,budget_list))

    ######################### Loading dataset  #########################
    data = Dataset(root='dataset', name=dataset_name)
    adj, features, labels = data.adj, data.features, data.labels

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


    rowlist = []
    test_model = 'GCN'

    for budget in budget_list:
        row = []
        target_node = select_nodes()

        #Orbit attack(1518)
        surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val)
        model = OrbitAttack(surrogate, df_orbit, nnodes=adj.shape[0], device=device)
        # model = OrbitAttackModified(surrogate, df_orbit, nnodes=adj.shape[0], device=device,attack_type="1518",orbit_type="two_Orbit_type")
        model = model.to(device)
        miss_percentage = attack(data,model, target_node,budget,features,adj,labels,test_model)
        row.append(miss_percentage)

        # Nettack
        surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val)
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        miss_percentage = attack_net(surrogate,data, target_node, budget, features, adj, labels,test_model)
        row.append(miss_percentage)

        '''
        Additional adversarial attacks can also be called easily with API provided by DeepRobust
        For further documentation: https://deeprobust.readthedocs.io/en/latest/
        For example, FGA can be called as follows:
        '''
        # #FGA
        # surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val)
        # model = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        # model = model.to(device)
        # miss_percentage = attack_net(surrogate, data, target_node, budget, features, adj, labels, test_model)
        # print(budget, miss_percentage)
        # row.append(miss_percentage)


        rowlist.append(row)

    #Save results
    result = pd.DataFrame(rowlist,columns=method,index=budget_list)
    result.to_csv('./results/result_{}_{}.csv'.format(test_model,dataset_name))
