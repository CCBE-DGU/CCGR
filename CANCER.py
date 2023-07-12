import torch
import pandas as pd
import numpy as np
import enum as enum
from torch_geometric.data import Data
from torch_geometric.data import Batch

BASE_EDGE = "Total Edge/totaledge_edge_"
BASE_NODE = "Total Node/totalnode_"

class ELabel(enum.Enum):
    BREAST_CANCER = (BASE_EDGE + "breast-cancer.csv", "breast-cancer")
    CANCER = (BASE_EDGE + "cancer.csv", "cancer")
    COLORECTAL_CANCER = (BASE_EDGE + "colorectal-cancer.csv", "colorectal-cancer")
    ENDOMETRIAL_CANCER = (BASE_EDGE + "endometrial-cancer.csv", "endometrial-cancer")
    GILOMA = (BASE_EDGE + "glioma.csv", "glioma")
    RENAL_CELL_CARCINOMA = (BASE_EDGE + "renal-cell-carcinoma.csv", "renal-cell-carcinoma")
    
    def __init__(self, filename, pathname):
        self.filename = filename
        self.pathname = pathname

class NLabel(enum.Enum):
    FULL = ("Total Node/FULL.csv")
    BRAC_TCGA_PAN_CAN_ATLAS = (BASE_NODE + "brca_tcga_pan_can_atlas_2018.csv")
    BRAC_TCGA_PUB2015 = (BASE_NODE + "brca_tcga_pub2015.csv")
    BRAC_TCGA = (BASE_NODE + "brca_tcga.csv")
    CCLE_BROAD_2019 = (BASE_NODE + "ccle_broad_2019.csv")
    COADREAD_TCGA_PAN_CAN_ATLAS_2018 = (BASE_NODE + "coadread_tcga_pan_can_atlas_2018.csv")
    KIRC_TCGA = (BASE_NODE + "kirc_tcga.csv")
    LGG_TCGA = (BASE_NODE + "lgg_tcga.csv")
    UCEC_TCGA_PAN_CAN_ATLAS = (BASE_NODE + "ucec_tcga_pan_can_atlas_2018.csv")

    def __init__(self, filename):
        self.filename = filename

class MyDataSet:
    def __init__(self, node_file, edge_files, detail = True, drop_y_threshold = 200, reverse=False, self_edge = False):
        self.node_df = None
        self.col2idx = {}
        self.idx2col = {}
        
        self.path_df = None
        
        self.edge_dfs = None
        self.eLabel2idx = None
        self.idx2eLabel = None
        
        self.y_df = None
        self.y2idx = {}
        self.idx2y = {}
        
        self.node_tensor = []
        self.path_list = []
        self.edge_tensor = None
        self.weights = None
        
        self.y = []
        
        self.__node_load(node_file, drop_y_threshold=drop_y_threshold, detail = detail)
        self.__edge_load(edge_files)
        self.__get_node()
        self.__get_edge(reverse, self_edge)

        
    def __node_load(self, node_files, drop_y_threshold, detail):
        for idx, file in enumerate(node_files):
            if idx == 0:
                self.node_df = pd.read_csv(file)
            else:
                self.node_df = pd.concat([self.node_df, pd.read_csv(file)], axis = 0)    

        na_df = self.node_df[self.node_df.isna().any(axis=1)]
        
        y_label = 'cancer_type_detailed' if detail else 'cancer_type'
        y_count = self.node_df[y_label].value_counts()
        drow_rows = y_count[y_count < drop_y_threshold].index
        self.node_df = self.node_df[~self.node_df[y_label].isin(drow_rows)]
        
        self.node_df = self.node_df.dropna(axis=0)
        self.node_df = self.node_df.reset_index(drop=True)
        self.col2idx = {col: idx for idx, col in enumerate(self.node_df.columns)}
        self.idx2col = {idx: col for idx, col in enumerate(self.node_df.columns)}

        self.y_df = self.node_df[[y_label]]
        y_list = [label[0] for label in self.y_df.values]
        self.y_set = set(y_list)
        print('yset ', len(self.y_set))
        for idx, label in enumerate(self.y_set):
            self.y2idx[label] = idx
            self.idx2y[idx] = label
        self.node_df.drop(['cancer_type', 'cancer_type_detailed'], axis=1, inplace=True)
        
        self.path_df = self.node_df['path']
        self.node_df.drop(labels='path', axis=1, inplace=True)
        
    def __edge_load(self, edge_files):
        self.eLabel2idx = {label.pathname: idx for idx, label in enumerate(edge_files)}
        self.idx2eLabel = {idx: label.pathname for idx, label in enumerate(edge_files)}
        self.edge_dfs = [pd.read_csv(edge.filename) for edge in edge_files]

    def __get_edge(self, reverse, self_edge):
        # TODO : tensor assign to tensor add
        self.weights = [[] for _ in range(len(self.edge_dfs))]
        self.edge_tensor = [[] for _ in range(len(self.edge_dfs))]
        
        for eidx, edge_df in enumerate(self.edge_dfs): 
            weight = []
            start_col = []
            end_col = []

            for idx, (s, e, w) in enumerate(edge_df.values):
                if w == 0:
                    continue
                start_col.append(self.col2idx[s])
                end_col.append(self.col2idx[e])
                if w > 0:
                    weight.append(1)
                else:
                    weight.append(-1)
            if self_edge:
                for i in range(self.node_tensor.shape[1]):
                    start_col.append(i)
                    end_col.append(i)
                    weight.append(1)

            self.weights[eidx] = torch.tensor(weight, dtype=torch.long) 
            if reverse:
                self.edge_tensor[eidx] = torch.tensor([end_col, start_col], dtype=torch.long)
            else:
                self.edge_tensor[eidx] = torch.tensor([start_col, end_col], dtype=torch.long)

    def __get_node(self):
        self.node_tensor = torch.tensor(self.node_df.values, dtype=torch.float32)
        
        self.node_tensor = self.node_tensor.reshape(-1, 176, 1)
        self.path_list = [self.eLabel2idx[eLabel] for eLabel in self.path_df.values]
        self.y = [self.y2idx[label[0]] for label in self.y_df.values]

    def make_dataset(self):
        data_list = []
        for i in range(len(self.node_tensor)):
            data = Data(
                        x = self.node_tensor[i],
                        edge_index = self.edge_tensor[self.path_list[i]],
                        edge_attr = self.weights[torch.tensor(self.path_list[i])],
                        y = torch.tensor(self.y[i], dtype=torch.long)
                    )
            data_list.append(data)
        dataset = Batch.from_data_list(data_list)
        dataset.num_classes = len(self.y_set)
        return dataset
    
    def make_dataset_onehot_column(self):
        data_list = []
        for i in range(len(self.node_tensor)):

            origin = self.node_tensor[i]
            new_node = torch.zeros((origin.shape[0], origin.shape[0]))
            for j in range(origin.shape[0]):
                new_node[j][j] += origin[j][0]
            
            data = Data(
                        x = new_node,
                        edge_index = self.edge_tensor[self.path_list[i]],
                        edge_attr = self.weights[torch.tensor(self.path_list[i])],
                        y = torch.tensor(self.y[i], dtype=torch.long)
                    )
            data_list.append(data)
        dataset = Batch.from_data_list(data_list)
        dataset.num_classes = len(self.y_set)
        return dataset
    
# test = MyDataSet(NLabel.BRAC_TCGA.filename, [ELabel.BREAST_CANCER])
# test_dataset = test.make_dataset_onehot_column()
# print(f'Dataset: {test_dataset}')
# print('-------------------')
# print(f'Number of graphs: {len(test_dataset)}')
# print(f'Number of nodes: {test_dataset[0].x.shape[0]}')
# print(f'Number of features: {test_dataset.num_features}')
# print(f'Numer of edge attr: {test_dataset.edge_attr}')
# print(f'Number of classes: { test_dataset.num_classes}')