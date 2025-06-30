import pickle as pkl
import numpy as np
import torch

def get_hypergraph(attrs):
    # 将数据转换为hypergraph
    """
    原理：
    首先，根据属性划分超边，即将每个节点的属性作为一个超边，组成超图；
    然后，根据得到的超图构建对应的邻接矩阵；
    接着，将邻接矩阵作为超图因子，增强图重构的能力；
    """
    # 获取节点数n和超边数e
    n, e = attrs.shape
    # 初始化超图邻接矩阵A
    A = np.zeros((n, n))
    # 遍历输入矩阵M的每一列
    for j in range(e):
        # 找到超边对应的节点组合
        nodes = np.where(attrs[:, j] == 1)[0]
        # 更新邻接矩阵A中对应位置的值
        for i in range(len(nodes)):
            for k in range(i + 1, len(nodes)):
                A[nodes[i], nodes[k]] += 1
                A[nodes[k], nodes[i]] += 1
    row_sums = A.sum(axis=1)
    # 行标准化
    A_row_normalized = A / (row_sums[:, np.newaxis] + 1)
    hyper_A = torch.FloatTensor(A_row_normalized)
    return hyper_A

def get_hypergraph_gpu(attrs):
    n, e = attrs.shape
    attrs_gpu = torch.FloatTensor(attrs).cuda()

    # Calculate the node combination of the hyperedge
    nodes = [torch.nonzero(attrs_gpu[:, j]).squeeze(1) for j in range(e)]
    A = torch.zeros(n, n).cuda()

    for edge_nodes in nodes:
        combinations = torch.combinations(edge_nodes, with_replacement=False)
        unique_combinations, counts = torch.unique(combinations, dim=0, return_counts=True)
        # Update the adjacency matrix
        A[unique_combinations[:, 0], unique_combinations[:, 1]] += counts
        A[unique_combinations[:, 1], unique_combinations[:, 0]] += counts
    row_sums = A.sum(dim=1)
    A_row_normalized = A / (row_sums.unsqueeze(1) + 1)
    hyper_A = A_row_normalized.cpu()
    # hyper_A = A.cpu()
    return hyper_A


if __name__ == '__main__':
    path = "./dataset/real_network/overlapping/mag_cs.pkl"
    with open(path, 'rb') as f:
        data = pkl.load(f)
        attrs = data['attr'].toarray()
        labels = data['label'].toarray()
        topo = data['topo'].toarray()
        print(attrs.shape, labels.shape, topo.shape)
    get_hypergraph(attrs)
    get_hypergraph_gpu(attrs)
