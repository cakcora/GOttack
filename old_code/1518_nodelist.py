import pandas as pd
import time
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from tqdm import tqdm
from deeprobust.graph.data import Dataset
from allfnc import  compute_logits


df_2d = pd.read_csv("../dataset/orbit/cora_orbit_table.csv")
device = "cpu"

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

def test1(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    acc_test = (output.argmax(1)[target_node] == labels[target_node])


    return acc_test.item()

data = Dataset(root='dataset', name='cora')


adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
idx_unlabeled_all = np.union1d(idx_unlabeled, idx_train)

######################### Setup Surrogate model  #########################

surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
nclass=labels.max().item()+1


degrees = adj.sum(0).A1

sum=0

for i in range(5):

    degrees = adj.sum(0).A1


    node_list = select_nodes() #[1554, 929, 2406, 1163, 1342, 1049, 2082, 1820, 1347, 1185, 1674, 132, 1765, 1306, 572, 10, 1314, 975, 1118, 591, 662, 805, 1605, 1463, 1196, 1255, 1526, 670, 1374, 668, 1481, 1515, 987, 1660, 7, 512, 2314, 1553, 2151, 251]
    num = len(node_list)

    miss= np.zeros((10, 5))

    edgemod= [1]

    false_class_node = []
    
    start_time= time.time()
     
    for gp in range(len(edgemod)):
        budget = edgemod[gp]
        cnt_gcn = 0
        cnt_gat = 0

        if type(adj) is torch.Tensor:
            ori_adj = utils.to_scipy(adj).tolil()
            lst_modified_adj = utils.to_scipy(adj).tolil()
            comp_modified_adj = utils.to_scipy(adj).tolil()  ########## for comparison only
            work_modified_adj = utils.to_scipy(
                adj).tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = utils.to_scipy(features).tolil()
            modified_features = utils.to_scipy(features).tolil()
        else:
            ori_adj = adj.tolil()
            modified_adj = adj.tolil()
            lst_modified_adj = adj.tolil()  ########## for comparison only
            work_modified_adj = adj.tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = features.tolil()
            modified_features = features.tolil()

        for target_node in tqdm(node_list):
            ############ Define the column name and the target value ########
            column_name = 'two_Orbit_type'
            Target_node = target_node

            # Find the indices of rows where the column has the target value
            matching_indices = df_2d.index[df_2d[column_name] == '1922'].tolist()


            perterbation_list = [[Target_node, value] for value in matching_indices]

            ########################### Using edge one by one  ##############################
            degrees = adj.sum(0).A1
            n_perturbations = budget  # int(degrees[target_node])

            for ep in range(n_perturbations):
                modified_adj_chk = lst_modified_adj.copy().tolil()
                ########### Getting score for each edge Addition/Deletion ##########

                edge_with_score = []

                ###### for one addition/deletion

                for i in perterbation_list:
                    modified_adj_chk_work = modified_adj_chk.copy().tolil()  ########## this will work as reset of the edge list
                    # #print(modified_adj_chk_work[i[0], i[1]])

                    modified_adj_chk_work[i[0], i[1]] = 1 - modified_adj_chk_work[i[0], i[1]]

                    logits_modified = compute_logits(modified_adj_chk_work, modified_features, surrogate, target_node)
                    label_u = labels[target_node]
                    label_target_onehot = np.eye(int(nclass))[labels[target_node]]
                    best_wrong_class = (logits_modified - 1000 * label_target_onehot).argmax()
                    surrogate_losses_modified = logits_modified[labels[target_node]] - logits_modified[best_wrong_class]
                    edge_with_score.append([i[0], i[1], surrogate_losses_modified])


                sorted_list = sorted(edge_with_score, key=lambda x: x[2])  # reverse=True)


                ################### final graph edge addition for testing ################

                final_modified_adj_chk = lst_modified_adj.copy().tolil()
                final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]] = 1 - final_modified_adj_chk[
                    sorted_list[0][0], sorted_list[0][1]]

                lst_modified_adj = final_modified_adj_chk.copy().tolil()

                list_of_lists = perterbation_list
                list_to_remove = [sorted_list[0][0], sorted_list[0][1]]
                filtered_list_of_lists = [x for x in list_of_lists if x != list_to_remove]

                perterbation_list = filtered_list_of_lists  # np.array(filtered_list_of_lists)


                if len(perterbation_list) == 0:
                    break


            acc_gcn = test1(final_modified_adj_chk, features, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            if acc_gcn == 0:
                cnt_gcn += 1


        miss[4][gp] = cnt_gcn / num

# Stop the timer
    end_time = time.time()

# Calculate the elapsed time
    elapsed_time = end_time - start_time


    sum= sum+elapsed_time

print(sum/5)
