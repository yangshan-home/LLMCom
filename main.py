import torch
import torch.nn.functional as F
from scipy import sparse
from methods.modules.improvgcn import ImprovGCN
from methods.modules.gcn import GCN
import dgl
import networkx as nx
from dataset.realnetwork import OverlappingRealNetwork, NonOverlappingRealNetwork, DyRealNetwork, DyRealNetworkEpinions
from dataset.synthetic import OverlappingSyntheticNetwork
import time
from metrics.overlapping import overlapping_nmi, overlapping_f1_score, find_best_thresh
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import numpy as np
from hypergraph import get_hypergraph, get_hypergraph_gpu
from sklearn.metrics import f1_score as fscore


class SMODGCN:
    def __init__(self, graph, num_class, data_by_llms=[[]], lr=5e-4, normalize_feature=False, lamda=0.5, alpha=1e-8,
                 beta=1e-6, flag_overlapping=False, llms_name=''):
        self.graph = graph
        self.hyper_adj_matrix = get_hypergraph_gpu(self.graph.ndata['feat'])
        self.adj = graph.adjacency_matrix().to_dense()
        self.sp_adj = sparse.csr_matrix(self.adj.numpy())
        self.num_class = num_class
        self.lr = lr
        self.normalize_feature = normalize_feature
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.B = self.compute_B_matrix()
        self.flag_overlapping = flag_overlapping
        self.data_by_llms = torch.tensor(data_by_llms)
        self.llms_name = llms_name

    def compute_B_matrix(self):
        nx_graph = dgl.to_networkx(self.graph)
        degree = nx.degree(nx_graph)
        deg = torch.FloatTensor([d for id, d in degree]).reshape(-1, 1)
        sum_deg = deg.sum()
        B = self.adj - (deg.matmul(deg.t()) / sum_deg)
        return B

    def l2_reg_loss(self, model, scale=1e-4):
        """Get L2 loss for model weights."""
        loss = 0.0
        for w in model.get_weights():
            loss = loss + w.pow(2.).sum()
        return loss * scale

    def l1_reg_loss(self, model, scale=1e-4):
        loss = 0.
        for w in model.get_weights():
            loss = loss + w.abs().sum()
        return loss * scale

    def train(self, max_epoch=300, unsupervised=False, loss_fun='CE', modul=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hyper_adj_matrix = self.hyper_adj_matrix.to(device)
        features = self.graph.ndata['feat'].to(device)

        in_feat = features.shape[1]  # 216
        n_hidden = 64
        num_class = self.num_class  # 27
        gcn = ImprovGCN(in_feat, n_hidden, num_class, 0., self.data_by_llms.shape[1], hyper_adj_matrix).to(device)

        optimizer = torch.optim.AdamW(gcn.parameters(), lr=self.lr)
        labels = self.graph.ndata['olp_label'].to(device)
        train_mask = self.graph.ndata['train_mask'].to(device)
        test_mask = self.graph.ndata['test_mask'].to(device)

        onmi_results = []
        nmi_results = []
        ari_results = []
        f1_score_results = []
        acc_results = []
        u_max = []

        for e in range(1, max_epoch + 1):
            gcn.train()
            u1, u2 = gcn(self.graph.to(device), features, self.data_by_llms.to(device), self.llms_name)
            u = u1.relu()
            hyper_out = u2

            mask = u >= 0.02
            u = torch.where(mask, u, torch.zeros_like(u))

            if unsupervised:
                loss = 0.
            else:
                loss = self.lamda * F.binary_cross_entropy(u[train_mask],
                                                           labels[train_mask].float())
            A_1 = torch.sigmoid(u.matmul(u.t()))  #
            A_0 = 1 - A_1
            A = self.adj.to(device) * torch.log(A_1) + (1 - self.adj.to(device)) * torch.log(A_0)
            loss_AGD = self.alpha * (-A.sum().sum())
            loss = loss + loss_AGD

            S = features.t()
            hyper_loss = F.mse_loss(S, hyper_out.t())
            loss = 0.5 * hyper_loss + loss


            if modul == True:
                loss_mm = self.beta * torch.trace(
                    u.t().matmul(self.B.to(device)).matmul(u))
                loss = loss - loss_mm

            loss = loss + self.l2_reg_loss(gcn, 1e-5)
            loss = loss + self.l1_reg_loss(gcn, 1e-6)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gcn.constraint()

            if self.flag_overlapping:
                if e % 1 == 0:
                    gcn.eval()
                    thresh = 3.0 / num_class
                    onmi = overlapping_nmi(labels[test_mask].cpu().numpy(), u[test_mask].detach().cpu().numpy(), thresh)
                    f1_score = overlapping_f1_score(labels[test_mask].cpu().numpy(), u[test_mask].detach().cpu().numpy(), thresh)

                    onmi_results.append(onmi)
                    f1_score_results.append(f1_score)
                    u_max.append(u[test_mask].detach().cpu().numpy())

            else:
                if e % 1 == 0:
                    gcn.eval()
                    real_labels = labels[test_mask].cpu().numpy().flatten()
                    temp_predict_labels = u[test_mask].detach().cpu().numpy()
                    predict_labels = np.zeros_like(temp_predict_labels, dtype=int)
                    max_indices = np.argmax(temp_predict_labels, axis=1)
                    predict_labels[np.arange(temp_predict_labels.shape[0]), max_indices] = 1
                    predict_labels_ = predict_labels.flatten()

                    temp_real_labels = real_labels.reshape(temp_predict_labels.shape)
                    real_labels_index = np.argmax(temp_real_labels, axis=1)
                    predict_labels_index = np.argmax(predict_labels, axis=1)
                    nmi = normalized_mutual_info_score(real_labels_index, predict_labels_index)
                    ari = adjusted_rand_score(real_labels, predict_labels_)
                    acc = accuracy_score(real_labels_index, predict_labels_index)

                    nmi_results.append(nmi)
                    ari_results.append(ari)
                    acc_results.append(acc)
                    u_max.append(u[test_mask].detach().cpu().numpy())

            if e == max_epoch:
                if self.flag_overlapping:
                    print('Max ONMI:{:.4f}'.format(np.max(onmi_results)))
                    print('Max Overlapping F1_Score:{:.4f}'.format(np.max(f1_score_results)))
                    print('Max Overlapping F1_Score epoch:{}'.format(np.argmax(f1_score_results) + 1))

                else:
                    print('Max NMI:{:.4f}'.format(np.max(nmi_results)))
                    print('Max ARI:{:.4f}'.format(np.max(ari_results)))
                    print('Max acc:{:.4f}'.format(np.max(acc_results)))
                    print('Max non-Overlapping ARI epoch:{}'.format(np.argmax(ari_results) + 1))


def non_overlapping_citeseer(llms_name):
    time_start = time.time()
    ds = NonOverlappingRealNetwork(path="./dataset/real_network/non-overlapping/citeseer.pkl",
                                   path_edges="./dataset/real_network/non-overlapping/citeseer_edges.dat",
                                   name="citeseer", mask_rate=[0.02, 0.0, 0.98], p_mis=0, TR=0)
    graph = ds[0]
    data_embedding_path_by_llms = f"./dataset/real_networks_by_llms/citeseer_nlp_embedding_by_{llms_name}.txt"
    with open(data_embedding_path_by_llms, 'r', encoding='utf-8') as f:
        data_lines = f.readlines()
        data_list = []
        for line in data_lines:
            str_list = line.split(' ')
            int_list = [float(item) for item in str_list]
            data_list.append(int_list)
    for i in range(20):
        model = SMODGCN(graph, num_class=ds.num_classes, lr=0.005, lamda=0.5, alpha=1e-9, beta=1e-6, data_by_llms=[[]], flag_overlapping=False, llms_name=llms_name)
        model.train(max_epoch=300)
        time_end = time.time()
        print('train time:', (time_end - time_start) // 60, 'min', (time_end - time_start) % 60, 's')


if __name__ == '__main__':
    """
    task1: The separation of overlapping and non-overlapping code is completed
    task2: Run experiments using non-overlapping datasets to complete the task
    task3: After adding llms, the overlapping and non-overlapping experiments were completed respectively
    task4: Some nodes have very long prompts and a large number of hops, resulting in insufficient context length for large models. This situation needs to be addressed. The selection is made by using the method of pagerank size, choosing the nodes with the highest 1-hop and 2-hop degrees. The purpose is to retain the most representative node information. Completed
    task5: How to ensure that large models can effectively learn the information of a long string of digital hops. Completed
    task6: During graph reconstruction, add a hypergraph reconstruction to achieve dual decoders: (1) Capture the high-order information of the graph to enhance the model's learning ability of the graph; Completed
    """
    # llms_name = 'llama31_70b_instruct'
    # llms_name = 'glm4_9b_chat'
    # llms_name = 'chatglm3_6b'
    llms_name = 'llama31_8b_instruct'
    # llms_name = 'qwen2_72b_instruct'

    non_overlapping_citeseer(llms_name)

