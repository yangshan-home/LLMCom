
import dgl
import numpy as np
from dgl.data import DGLDataset
import pandas as pd
import random
import torch
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')


class OverlappingRealNetwork(DGLDataset):
    def __init__(self, path="", path_edges="", name="", mask_shuffle=True, self_loop=True, mask_rate=[0.2, 0.6, 0.2], p_mis=0., TR=0):
        # if not path.endswith("/"):
        #     path += "/"
        self.path = path
        self.path_edges = path_edges
        self.mask_shuffle = mask_shuffle
        self.self_loop = self_loop
        self.mask_rate = mask_rate
        self.p_mis = p_mis
        self.TR = TR
        # self.num_community = self.get_num_community(self.path + 'community.dat')
        assert sum(mask_rate) == 1
        super(OverlappingRealNetwork, self).__init__(name=name)

    def process(self):
        path = self.path
        path_edges = self.path_edges
        # features = np.loadtxt(fname=path + "features.dat", dtype=np.float32, delimiter='\t')
        # labels = np.loadtxt(path + 'community.dat')
        
        with open(path, 'rb') as f:
            data = pkl.load(f) # data['...']
        labels = data['label'].toarray()  
        features = data['attr'].toarray() 
        edges = np.loadtxt(path_edges, dtype=np.int32)
        
        self.num_node = labels.shape[0]
        num_nodes = self.num_node
        self.num_community = labels.shape[1]

        mask_index = np.arange(start=0, stop=num_nodes)
        if self.mask_shuffle:
            np.random.shuffle(mask_index)

        train_mask = torch.BoolTensor(np.zeros(num_nodes))
        val_mask = torch.BoolTensor(np.zeros(num_nodes))
        test_mask = torch.BoolTensor(np.zeros(num_nodes))

        for i, v in enumerate(mask_index):
            if i < num_nodes * self.mask_rate[0]:
                train_mask[v] = True
            elif i < num_nodes * (self.mask_rate[0] + self.mask_rate[1]):
                # print("sss", num_nodes * self.mask_rate[0] + self.mask_rate[1])
                val_mask[v] = True
            else:
                test_mask[v] = True

        # Handle edge files
        src = edges[:, 0]
        dest = edges[:, 1]

        # Exchange node topology
        num_topo = int(num_nodes * self.TR / 2)
        for i in range(num_topo):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            if a == b:
                continue
            else:
                indices_a_src = np.where(src == a)[0]
                indices_b_src = np.where(src == b)[0]
                indices_a_dest = np.where(dest == a)[0]
                indices_b_dest = np.where(dest == b)[0]
                for ii in indices_a_src:
                    src[ii] = b
                for iii in indices_a_dest:
                    dest[iii] = b
                for jj in indices_b_src:
                    src[jj] = a
                for jjj in indices_b_dest:
                    dest[jjj] = a

        #  Exchange node attributes
        swap_count = int(num_nodes * self.p_mis / 2)
        for i in range(swap_count):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            # print(a, ' ', b)
            if a == b:
                continue
            features[[a, b], :] = features[[b, a], :]

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        self.graph = dgl.graph((src, dest), num_nodes=num_nodes)
        # if self.self_loop:
        #     self.graph = dgl.add_self_loop(self.graph)
        
        self.graph.ndata['feat'] = features
        self.graph.ndata['olp_label'] = labels
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        print("node number:", self.graph.ndata['olp_label'].shape[0], 'community number:', self.graph.ndata['olp_label'].shape[1],
              "attr number:",
              self.graph.ndata['feat'].shape[1], "edge number:", self.graph.edges()[0].shape[0])

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.graph

    @property
    def num_classes(self):
        return self.num_community


class NonOverlappingRealNetwork(DGLDataset):
    def __init__(self, path="", path_edges="", name="", mask_shuffle=True, self_loop=True, mask_rate=[0.2, 0.6, 0.2],
                 p_mis=0., TR=0):
        # if not path.endswith("/"):
        #     path += "/"
        self.path = path
        self.path_edges = path_edges
        self.mask_shuffle = mask_shuffle
        self.self_loop = self_loop
        self.mask_rate = mask_rate
        self.p_mis = p_mis
        self.TR = TR
        assert sum(mask_rate) == 1
        super(NonOverlappingRealNetwork, self).__init__(name=name)

    def process(self):
        path = self.path
        path_edges = self.path_edges

        with open(path, 'rb') as f:
            data = pkl.load(f)  # data['...']
        labels_matrix = data['label']
        nn = labels_matrix.shape[0]
        labels = np.zeros((nn, np.max(data['label'])+1), dtype=int)
        labels[np.arange(nn), labels_matrix] = 1

        features = data['attr'].toarray()
        edges = np.loadtxt(path_edges, dtype=np.int32)

        self.num_node = labels.shape[0]
        num_nodes = self.num_node
        self.num_community = labels.shape[1]

        mask_index = np.arange(start=0, stop=num_nodes)
        if self.mask_shuffle:
            np.random.shuffle(mask_index)

        train_mask = torch.BoolTensor(np.zeros(num_nodes))
        val_mask = torch.BoolTensor(np.zeros(num_nodes))
        test_mask = torch.BoolTensor(np.zeros(num_nodes))

        for i, v in enumerate(mask_index):
            if i < num_nodes * self.mask_rate[0]:
                train_mask[v] = True
            elif i < num_nodes * (self.mask_rate[0] + self.mask_rate[1]):
                # print("sss", num_nodes * self.mask_rate[0] + self.mask_rate[1])
                val_mask[v] = True
            else:
                test_mask[v] = True

        src = edges[:, 0]
        dest = edges[:, 1]

        num_topo = int(num_nodes * self.TR / 2)
        for i in range(num_topo):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            if a == b:
                continue
            else:
                indices_a_src = np.where(src == a)[0]
                indices_b_src = np.where(src == b)[0]
                indices_a_dest = np.where(dest == a)[0]
                indices_b_dest = np.where(dest == b)[0]
                for ii in indices_a_src:
                    src[ii] = b
                for iii in indices_a_dest:
                    dest[iii] = b
                for jj in indices_b_src:
                    src[jj] = a
                for jjj in indices_b_dest:
                    dest[jjj] = a

        #  Exchange node attributes
        swap_count = int(num_nodes * self.p_mis / 2)
        for i in range(swap_count):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            # print(a, ' ', b)
            if a == b:
                continue
            features[[a, b], :] = features[[b, a], :]

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        self.graph = dgl.graph((src, dest), num_nodes=num_nodes)
        # if self.self_loop:
        #     self.graph = dgl.add_self_loop(self.graph)

        self.graph.ndata['feat'] = features
        self.graph.ndata['olp_label'] = labels
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        # self.graph.ndata['num_classes'] = self.num_community
        print("node number:", self.graph.ndata['olp_label'].shape[0], 'community number:', self.graph.ndata['olp_label'].shape[1],
              "attr number:",
              self.graph.ndata['feat'].shape[1], "edge number:", self.graph.edges()[0].shape[0])
    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.graph

    @property
    def num_classes(self):
        return self.num_community


class DyRealNetwork(DGLDataset):
    def __init__(self, path="", path_edges="", name="", shot_num=0, mask_shuffle=True, self_loop=True, mask_rate=[0.2, 0.6, 0.2],
                 p_mis=0., TR=0):
        # if not path.endswith("/"):
        #     path += "/"
        self.path = path
        self.path_edges = path_edges
        self.mask_shuffle = mask_shuffle
        self.self_loop = self_loop
        self.mask_rate = mask_rate
        self.p_mis = p_mis
        self.TR = TR
        self.shot_num = shot_num
        assert sum(mask_rate) == 1
        super(DyRealNetwork, self).__init__(name=name)

    def process(self):
        path = self.path
        path_edges = self.path_edges

        with open(path, 'rb') as f:
            data = pkl.load(f)  # data['...']
        labels_matrix = data['label'][self.shot_num]
        nn = labels_matrix.shape[0]
        labels = np.zeros((nn, np.max(data['label'])+1), dtype=int)
        labels[np.arange(nn), labels_matrix] = 1

        features = data['feat'].toarray()
        edges = np.loadtxt(path_edges, dtype=np.int32)

        self.num_node = labels.shape[0]
        num_nodes = self.num_node
        self.num_community = labels.shape[1]

        mask_index = np.arange(start=0, stop=num_nodes)
        if self.mask_shuffle:
            np.random.shuffle(mask_index)

        train_mask = torch.BoolTensor(np.zeros(num_nodes))
        val_mask = torch.BoolTensor(np.zeros(num_nodes))
        test_mask = torch.BoolTensor(np.zeros(num_nodes))

        for i, v in enumerate(mask_index):
            if i < num_nodes * self.mask_rate[0]:
                train_mask[v] = True
            elif i < num_nodes * (self.mask_rate[0] + self.mask_rate[1]):
                # print("sss", num_nodes * self.mask_rate[0] + self.mask_rate[1])
                val_mask[v] = True
            else:
                test_mask[v] = True

        # Handle edge files
        src = edges[:, 0]
        dest = edges[:, 1]

        # Exchange node topology
        num_topo = int(num_nodes * self.TR / 2)
        for i in range(num_topo):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            if a == b:
                continue
            else:
                indices_a_src = np.where(src == a)[0]
                indices_b_src = np.where(src == b)[0]
                indices_a_dest = np.where(dest == a)[0]
                indices_b_dest = np.where(dest == b)[0]
                for ii in indices_a_src:
                    src[ii] = b
                for iii in indices_a_dest:
                    dest[iii] = b
                for jj in indices_b_src:
                    src[jj] = a
                for jjj in indices_b_dest:
                    dest[jjj] = a

        #  Exchange node attributes
        swap_count = int(num_nodes * self.p_mis / 2)
        for i in range(swap_count):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            # print(a, ' ', b)
            if a == b:
                continue
            features[[a, b], :] = features[[b, a], :]

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        self.graph = dgl.graph((src, dest), num_nodes=num_nodes)
        if self.self_loop:
            self.graph = dgl.add_self_loop(self.graph)

        self.graph.ndata['feat'] = features
        self.graph.ndata['olp_label'] = labels
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        # self.graph.ndata['num_classes'] = self.num_community
        print("node number:", self.graph.ndata['olp_label'].shape[0], 'community number:', self.graph.ndata['olp_label'].shape[1],
              "attr number:",
              self.graph.ndata['feat'].shape[1], "edge number:", self.graph.edges()[0].shape[0])

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.graph

    @property
    def num_classes(self):
        return self.num_community


class DyRealNetworkEpinions(DGLDataset):
    def __init__(self, path="", path_edges="", name="", shot_num=0, mask_shuffle=True, self_loop=True, mask_rate=[0.2, 0.6, 0.2],
                 p_mis=0., TR=0):
        # if not path.endswith("/"):
        #     path += "/"
        self.path_label = path
        self.path_edges = path_edges
        self.mask_shuffle = mask_shuffle
        self.self_loop = self_loop
        self.mask_rate = mask_rate
        self.p_mis = p_mis
        self.TR = TR
        self.shot_num = shot_num
        assert sum(mask_rate) == 1
        super(DyRealNetworkEpinions, self).__init__(name=name)

    def process(self):
        labels_matrix = np.loadtxt(self.path_label, dtype=np.int32)
        num_nodes = labels_matrix.shape[0]
        edges = np.loadtxt(self.path_edges, dtype=np.int32)
        feature_path = '/T20050027/ShanYang/paper_second/paper2/dataset/dynamic networks/epinions_with_timestamps_11/epinion_timestamp_features.dat'
        feature_matrix_ = np.loadtxt(feature_path, dtype=np.int32)
        feature_matrix = feature_matrix_[:num_nodes, :]

        assert num_nodes == len(np.unique(edges.flatten()))
        self.num_community = labels_matrix.shape[1]

        mask_index = np.arange(start=0, stop=num_nodes)
        if self.mask_shuffle:
            np.random.shuffle(mask_index)

        train_mask = torch.BoolTensor(np.zeros(num_nodes))
        val_mask = torch.BoolTensor(np.zeros(num_nodes))
        test_mask = torch.BoolTensor(np.zeros(num_nodes))

        for i, v in enumerate(mask_index):
            if i < num_nodes * self.mask_rate[0]:
                train_mask[v] = True
            elif i < num_nodes * (self.mask_rate[0] + self.mask_rate[1]):
                # print("sss", num_nodes * self.mask_rate[0] + self.mask_rate[1])
                val_mask[v] = True
            else:
                test_mask[v] = True

        # Handle edge files
        src = edges[:, 0]
        dest = edges[:, 1]


        features = torch.FloatTensor(feature_matrix)
        labels = torch.LongTensor(labels_matrix)

        self.graph = dgl.graph((src, dest), num_nodes=num_nodes)
        if self.self_loop:
            self.graph = dgl.add_self_loop(self.graph)

        self.graph.ndata['feat'] = features
        self.graph.ndata['olp_label'] = labels
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        # self.graph.ndata['num_classes'] = self.num_community
        print("node number:", self.graph.ndata['olp_label'].shape[0], 'community number:', self.graph.ndata['olp_label'].shape[1],
              "attr number:",
              self.graph.ndata['feat'].shape[1], "edge number:", self.graph.edges()[0].shape[0])

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.graph

    @property
    def num_classes(self):
        return self.num_community




def example_fb_1912():
    fb_1912 = OverlappingRealNetwork(path="/home/slcheng/yangshan/nural network/NEW_SSGCAE/dataset/real_network/overlapping/fb_1912.pkl", 
                                     path_edges="/home/slcheng/yangshan/nural network/NEW_SSGCAE/dataset/real_network/overlapping/fb_1912_edges.dat", name="fb_1912")
    graph = fb_1912[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

def example_fb_1684():
    fb_1684 = OverlappingRealNetwork(path="./real_network/overlapping/fb_1684.pkl",
                                     path_edges="./real_network/overlapping/fb_1684_edges.dat", name="fb_1684")
    graph = fb_1684[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

def example_mag_cs():
    fb_1684 = OverlappingRealNetwork(path="/home/slcheng/yangshan/nural network/NEW_SSGCAE/dataset/real_network/overlapping/mag_cs.pkl", 
                                     path_edges="/home/slcheng/yangshan/nural network/NEW_SSGCAE/dataset/real_network/overlapping/mag_cs_edges.dat", name="mag_cs")
    graph = fb_1684[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

def example_mag_eng():
    mag_eng = OverlappingRealNetwork(path="/home/slcheng/yangshan/nural network/NEW_SSGCAE/dataset/real_network/overlapping/mag_eng.pkl", 
                                     path_edges="/home/slcheng/yangshan/nural network/NEW_SSGCAE/dataset/real_network/overlapping/mag_eng_edges.dat", name="mag_eng")
    graph = mag_eng[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

def non_overlapping_citeseer():
    citeseer = NonOverlappingRealNetwork(path="./real_network/non-overlapping/citeseer.pkl",
                                     path_edges="./real_network/non-overlapping/citeseer_edges.dat", name="citeseer")
    graph = citeseer[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

def non_overlapping_cora():
    citeseer = NonOverlappingRealNetwork(path="./real_network/non-overlapping/cora.pkl",
                                     path_edges="./real_network/non-overlapping/cora_edges.dat", name="cora")
    graph = citeseer[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

def non_overlapping_pubmed():
    citeseer = NonOverlappingRealNetwork(path="./real_network/non-overlapping/pubmed.pkl",
                                     path_edges="./real_network/non-overlapping/pubmed_edges.dat", name="pubmed")
    graph = citeseer[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

def non_overlapping_flickr():
    citeseer = NonOverlappingRealNetwork(path="./real_network/non-overlapping/flickr.pkl",
                                     path_edges="./real_network/non-overlapping/flickr_edges.dat", name="flickr")
    graph = citeseer[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)
def non_overlapping_blogcatalog():
    blogcatalog = NonOverlappingRealNetwork(path="./real_network/non-overlapping/blogcatalog.pkl",
                                         path_edges="./real_network/non-overlapping/blogcatalog_edges.dat", name="flickr")
    graph = blogcatalog[0]
    print("Number of nodes:", graph.ndata['olp_label'].shape[0], 'Number of communities:', graph.ndata['olp_label'].shape[1], "Attribute number:",
          graph.ndata['feat'].shape[1])
    print("Label dimension:", graph.ndata['olp_label'].shape)
    print("Feature dimension:", graph.ndata['feat'].shape)
    print('Training set dimension:', graph.ndata['train_mask'].shape)
    print('Verification set dimension:', graph.ndata['val_mask'].shape)
    print('Test set dimension:', graph.ndata['test_mask'].shape)

if __name__ == '__main__':
    # example_fb_1912()
    # example_fb_1684()
    # example_mag_cs()
    # example_mag_eng()
    # non_overlapping_citeseer()
    # non_overlapping_cora()
    # non_overlapping_pubmed()
    # non_overlapping_flickr()
    non_overlapping_blogcatalog()