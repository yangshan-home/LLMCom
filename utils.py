import pickle as pkl
import numpy as np
import scipy.io
import pandas as pd
from collections import defaultdict
import random

def get_hop_neighbors(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    result = []
    # 去除自环
    adj_matrix = np.where(np.eye(num_nodes) == 1, 0, adj_matrix)
    # print(adj_matrix)
    for node in range(num_nodes):
        hop_0 = {node}
        hop_1 = set(np.where(adj_matrix[node] == 1)[0])
        hop_2 = set()
        for neighbor in hop_1:
            hop_2.update(np.where(adj_matrix[neighbor] == 1)[0])
        # 去除重复的节点
        hop_2 -= hop_1
        hop_2 -= hop_0
        result.append((hop_0, hop_1, hop_2))
    return result


def hop(adj_matrix, name='temp'):
    """
    hop-0:表示节点的自身
    hop-1:表示节点的邻居节点集合
    hop-2:表示节点的二阶邻居节点集合
    return: 每个节点的hop-0是什么，hop-1是什么，hop-2是什么。
    """

    result = get_hop_neighbors(adj_matrix)
    with open(f"/T20050027/ShanYang/paper_second/paper2/dataset/real_networks_by_llms/{name}_nlp.txt", 'w', encoding='utf-8') as f:
        for i, (hop_0, hop_1, hop_2) in enumerate(result):
            # pre_result = f"The 0-hop, 1-hop, and 2-hop of node {i} are {hop_0}, {hop_1}, and {hop_2}, respectively."
            pre_result = f"As a network structure analyst, please analyze and study the topology of the entire network based on the adjacency information of the following nodes. The adjacency information for each node includes 0-hop, 1-hop, and 2-hop." \
                         f"Note that 0-hop refers to the node itself, 1-hop refers to directly connected neighbor nodes, and 2-hop refers to nodes that are directly connected to the 1-hop neighbor nodes." \
                         f"The following is the information for node {i}:" \
                         f"Adjacency information for node {i}: 0-hop: {hop_0} (the node itself) " \
                         f"1-hop: {hop_1} (directly connected neighbors) " \
                         f"2-hop: {hop_2} (nodes directly connected to 1-hop neighbors) " \
                         f"Based on this information, please: " \
                         f"1. Understand the position and role of each node within the network. " \
                         f"2. Analyze how the connections at different levels influence the overall structure of the network. " \
                         f"3. Learn and generate an embedding representation that effectively captures the relationships between nodes and the topological features of the network."
            print(f"第{i+1}个节点构造完成！")
            f.write(pre_result + '\n')


def example_fb_1684():
    path_network = './dataset/real_network/overlapping/fb_1684.pkl'
    path_edges = "./dataset/real_network/overlapping/fb_1684_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    labels = data['label'].toarray()
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = labels.shape[0]
    num_community = labels.shape[1]
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="fb_1684")

def example_fb_1912():
    path_network = './dataset/real_network/overlapping/fb_1912.pkl'
    path_edges = "./dataset/real_network/overlapping/fb_1912_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    labels = data['label'].toarray()
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = labels.shape[0]
    num_community = labels.shape[1]
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="fb_1912")

def example_mag_eng():
    path_network = './dataset/real_network/overlapping/mag_eng.pkl'
    path_edges = "./dataset/real_network/overlapping/mag_eng_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    labels = data['label'].toarray()
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = labels.shape[0]
    num_community = labels.shape[1]
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="mag_eng")

def example_mag_cs():
    path_network = './dataset/real_network/overlapping/mag_cs.pkl'
    path_edges = "./dataset/real_network/overlapping/mag_cs_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    labels = data['label'].toarray()
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = labels.shape[0]
    num_community = labels.shape[1]
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="mag_cs")


def non_overlapping_citeseer():
    path_network = './dataset/real_network/non-overlapping/citeseer.pkl'
    path_edges = "./dataset/real_network/non-overlapping/citeseer_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = features.shape[0]
    num_community = np.max(data['label'])+1
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="citeseer")

def non_overlapping_cora():
    path_network = './dataset/real_network/non-overlapping/cora.pkl'
    path_edges = "./dataset/real_network/non-overlapping/cora_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = features.shape[0]
    num_community = np.max(data['label'])+1
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="cora")

def non_overlapping_pubmed():
    path_network = './dataset/real_network/non-overlapping/pubmed.pkl'
    path_edges = "./dataset/real_network/non-overlapping/pubmed_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = features.shape[0]
    num_community = np.max(data['label'])+1
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="pubmed")

def non_overlapping_flickr():
    path_network = './dataset/real_network/non-overlapping/flickr.pkl'
    path_edges = "./dataset/real_network/non-overlapping/flickr_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = features.shape[0]
    num_community = np.max(data['label'])+1
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="flickr")


def non_overlapping_blogcatalog():
    path_network = './dataset/real_network/non-overlapping/blogcatalog.pkl'
    path_edges = "./dataset/real_network/non-overlapping/blogcatalog_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    features = data['attr'].toarray()
    topology = data['topo'].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = features.shape[0]
    num_community = np.max(data['label'])+1
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name="blogcatalog")


def DyCora_Snapshots():
    shot_num = 4
    path_network = '/T20050027/ShanYang/paper_second/paper2/dataset/dynamic networks/DyCora.pkl'
    path_edges = f"/T20050027/ShanYang/paper_second/paper2/dataset/dynamic networks/DyCora_shot{shot_num}_edges.dat"
    with open(path_network, 'rb') as f:
        data = pkl.load(f)
    labels = data['label'][shot_num]
    features = data['feat'].toarray()
    topology = data['adj'][shot_num].toarray()
    edges = np.loadtxt(path_edges, dtype=np.int32)
    num_node = features.shape[0]
    num_community = len(set(labels))
    print("节点数：", num_node, "边数：", edges.shape[0], '社区数：', num_community, "属性数：", features.shape[1])
    # print("拓扑结构：", topology)
    hop(topology, name=f"DyCora_shot{shot_num}")


if __name__ == '__main__':
    non_overlapping_citeseer()
