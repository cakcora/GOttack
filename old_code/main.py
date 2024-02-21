import pandas as pd
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph import utils
from tqdm import tqdm
from model.GIN import GIN
from model.GSAGE import GraphSAGE
from allfnc import compute_logits

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def select_node(target_gcn=None):

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
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
    high = [x for x, y in sorted_margins[: 1]]
    low = [x for x, y in sorted_margins[-1: ]]
    other = [x for x, y in sorted_margins[1: -1]]
    other = np.random.choice(other, 2, replace=False).tolist()

    return high + low + other

def test_GCN(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    #probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    #print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    #print(output.argmax(1)[target_node])
    #print(labels[target_node])
    return acc_test.item()

def test_GIN(adj, data, target_node):
    ''' test on GIN '''
    pyg_data = Dpr2Pyg(data)
    gin = GIN(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gin = gin.to(device)
    perturbed_adj = adj.tocsr()           ###########   perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    gin.fit(pyg_data, verbose=False)  ############ train with earlystopping
    gin.eval()
    output = gin.predict()
    #probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    #print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    #print(output.argmax(1)[target_node])
    #print(labels[target_node])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test.item()



def test_GSAGE(adj, data, target_node):
    ''' test on GSAGE '''
    pyg_data = Dpr2Pyg(data)
    gsage = GraphSAGE(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gsage = gsage.to(device)
    perturbed_adj = adj.tocsr()           ###########   perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    gsage.fit(pyg_data, verbose=False)
    gsage.eval()
    output = gsage.predict()
    #probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    #print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    #print(output.argmax(1)[target_node])
    #print(labels[target_node])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test.item()



####### Read the Graphlet data from the file

with open('../dataset/orbit/dpr_polblogs.out', 'r') as file:
    lines = file.readlines()

# Process the lines to create a list of lists
data_list = [list(map(int, line.split())) for line in lines]

# Convert the list of lists to a NumPy array

graphlet_features = np.array(data_list)
print(graphlet_features)


mylist = []
for i in range(len(graphlet_features)):
    arr = graphlet_features[i]
    print(arr)
    sorted_indices = np.argsort(arr)[::-1]
    print(sorted_indices)

    if sorted_indices[0] < sorted_indices[1]:
        s1 = str(sorted_indices[0]) + str(sorted_indices[1])
    else:
        s1 = str(sorted_indices[1]) + str(sorted_indices[0])

    mylist.append([i, sorted_indices[0], sorted_indices[1], s1])

my_array = np.array(mylist)
df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
print(df_2d)

######################### Loding the dataset   #########################
data = Dataset(root='', name='polblogs')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

######################### Setup Surrogate model  #########################
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

nclass=labels.max().item()+1

filenames = ['newfile.csv']
# Assuming you have a loop where you generate data for each CSV file
for filename in filenames:

    degrees = adj.sum(0).A1
    node_list = select_node()
    num = len(node_list)
    print(node_list)
    rmv_del_edge=[]
    miss= np.zeros((9, 5))

    edgemod= [1] #, 2, 3, 4, 5]

    false_class_node = []

    for gp in range(len(edgemod)):
        budget = edgemod[gp]
        cnt_gcn = 0
        cnt_gat = 0
        if type(adj) is torch.Tensor:
            ori_adj = utils.to_scipy(adj).tolil()
            lst_modified_adj = utils.to_scipy(adj).tolil()
            comp_modified_adj = utils.to_scipy(adj).tolil()  ########## for comparison only
            work_modified_adj = utils.to_scipy(adj).tolil()  ######## for working menas modify it by adding/removing edges
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
            Target_node_orbit = df_2d.iloc[Target_node, 3]
            # Find the indices of rows where the column has the target value
            matching_indices = df_2d.index[df_2d[column_name] == Target_node_orbit].tolist()
            perterbation_list = [[Target_node, value] for value in matching_indices]

            n_perturbations= budget

            for ep in range(n_perturbations):
                modified_adj_chk = lst_modified_adj.copy().tolil()
                ########### Getting score for each edge Addition/Deletion ##########

                edge_with_score = []

                ###### for one addition/deletion

                for i in perterbation_list:
                    modified_adj_chk_work = modified_adj_chk.copy().tolil()  ########## this will work as reset of the edge list
                    # print(modified_adj_chk_work[i[0], i[1]])

                    modified_adj_chk_work[i[0], i[1]] = 1 - modified_adj_chk_work[i[0], i[1]]

                    logits_modified = compute_logits(modified_adj_chk_work, modified_features, surrogate, target_node)
                    label_u = labels[target_node]
                    label_target_onehot = np.eye(int(nclass))[labels[target_node]]
                    best_wrong_class = (logits_modified - 1000 * label_target_onehot).argmax()
                    surrogate_losses_modified = logits_modified[labels[target_node]] - logits_modified[best_wrong_class]
                    edge_with_score.append([i[0], i[1], surrogate_losses_modified])

                sorted_list = sorted(edge_with_score, key=lambda x: x[2])

                ################### final graph edge addition for testing ################

                final_modified_adj_chk = lst_modified_adj.copy().tolil()
                final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]] = 1 - final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]]

                lst_modified_adj = final_modified_adj_chk.copy().tolil()
                list_of_lists = perterbation_list
                list_to_remove = [sorted_list[0][0], sorted_list[0][1]]
                filtered_list_of_lists= [x for x in list_of_lists if x != list_to_remove]
                perterbation_list = filtered_list_of_lists

                if len(perterbation_list) == 0:
                    break
                #print(len(filtered_list_of_lists))

            # acc_gcn = test_GCN(final_modified_adj_chk, features, target_node)
            # print('class : %s' % acc_gcn)
            # if acc_gcn == 0:
            # cnt_gcn += 1

            acc_gat = test_GIN(final_modified_adj_chk, data, target_node)
            print('class : %s' %acc_gat)
            if acc_gat == 0:
              cnt_gat += 1
        print('========= TESTING Same Orbit Connection =========')
        print('Missclassification rate for Same Orbit Connection : %s' % (cnt_gat / num))

        miss[0][gp]= cnt_gat / num

######################## Nettack  #####################

        cnt_gcn = 0
        cnt_gat = 0
        degrees = adj.sum(0).A1

        for target_node in tqdm(node_list):

            n_perturbations= budget

            model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
            model = model.to(device)
            model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
            modified_adj = model.modified_adj

            #acc_gcn = test_GCN(modified_adj, features, target_node)
            #print('class : %s' % acc_gcn)
            #if acc_gcn == 0:
               # cnt_gcn += 1

            acc_gat = test_GIN(modified_adj, data, target_node)
            print('class : %s' % acc_gat)
            if acc_gat == 0:
                cnt_gat += 1

        print('=== Testing Nettack  ===')

        print('Miss-classification rate Nettack: %s' % (cnt_gat / num))

        miss[2][gp] = cnt_gat / num


##################################### 1518 Not same orbit connection ##############################

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
            matching_indices = df_2d.index[df_2d[column_name] == '1518'].tolist()

            perterbation_list = [[Target_node, value] for value in matching_indices]

            n_perturbations = budget

            for ep in range(n_perturbations):
                modified_adj_chk = lst_modified_adj.copy().tolil()
                ########### Getting score for each edge Addition/Deletion ##########

                edge_with_score = []

                for i in perterbation_list:
                    modified_adj_chk_work = modified_adj_chk.copy().tolil()  ########## this will work as reset of the edge list
                    # print(modified_adj_chk_work[i[0], i[1]])

                    modified_adj_chk_work[i[0], i[1]] = 1 - modified_adj_chk_work[i[0], i[1]]

                    logits_modified = compute_logits(modified_adj_chk_work, modified_features, surrogate, target_node)
                    label_u = labels[target_node]
                    label_target_onehot = np.eye(int(nclass))[labels[target_node]]
                    best_wrong_class = (logits_modified - 1000 * label_target_onehot).argmax()
                    surrogate_losses_modified = logits_modified[labels[target_node]] - logits_modified[best_wrong_class]
                    edge_with_score.append([i[0], i[1], surrogate_losses_modified])

                sorted_list = sorted(edge_with_score, key=lambda x: x[2])

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

            # acc_gcn = test_GCN(modified_adj, features, target_node)
            # print('class : %s' % acc_gcn)
            # if acc_gcn == 0:
            # cnt_gcn += 1

            acc_gat = test_GIN(final_modified_adj_chk, data, target_node) #single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' %acc_gat)

            if target_node in false_class_node:
                cnt_gat += 1
            elif acc_gat == 0:
                cnt_gat += 1
                false_class_node.append(target_node)

        print('========= TESTING with 1518 nodes =========')

        print('Missclassification rate for 1518 : %s' % (cnt_gat / num))

        miss[4][gp] = cnt_gat / num




print(miss)
df = pd.DataFrame(miss)
df.to_csv(filename, index=False)