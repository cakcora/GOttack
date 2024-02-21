import os
import numpy as np
import pandas as pd


class OrbitTableGenerator:
    def __init__(self,dataset):
        self.dataset = dataset
        self.filepath = './dataset/orbit/'

    def generate_orbit_table(self):
        if self.dataset == 'cora':
            return self.generate_orbit_tables_from_sratch()
        elif self.dataset == 'citeseer':
            return self.generate_orbit_tables_from_sratch()
        elif self.dataset == 'polblogs':
            return self.generate_orbit_tables_from_sratch()
        else:
            raise Exception("Unsupport dataset")

    def generate_orbit_tables_from_sratch(self):
        filename = self.filepath +"dpr_"+ self.dataset + '.out'
        with open(filename, 'r') as file:
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
        return df_2d