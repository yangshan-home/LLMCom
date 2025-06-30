import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import torch.nn as nn
import torch, math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MultiHead_SelfAttention2(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads, hidden_dropout_prob):
        '''
        Args:
            dim: dimension for each time step
            num_head:num head for multi-head self-attention
        '''
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        input_tensor = torch.unsqueeze(input_tensor, 0)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # print('hidden_states:', hidden_states.shape)
        hidden_states = torch.squeeze(hidden_states, 0)
        return hidden_states


class GroupQueryAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads, hidden_dropout_prob, num_groups):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.num_groups = num_groups

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        input_tensor = torch.unsqueeze(input_tensor, 0)  # Add batch dimension
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Group Query Attention: We will compute attention for each group separately
        batch_size, num_heads, seq_length, head_size = query_layer.size()
        group_size = seq_length // self.num_groups

        # Reshape query, key, and value for groups
        query_layer = query_layer.view(batch_size, num_heads, self.num_groups, group_size, head_size)
        key_layer = key_layer.view(batch_size, num_heads, self.num_groups, group_size, head_size)
        value_layer = value_layer.view(batch_size, num_heads, self.num_groups, group_size, head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (B, H, G, Sg, Sg)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (B, H, G, Sg, Hd)
        context_layer = context_layer.view(batch_size, num_heads, seq_length, self.attention_head_size)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = torch.squeeze(hidden_states, 0)  # Remove batch dimension
        return hidden_states


class CRD(torch.nn.Module):
    def __init__(self, in_feats, n_hid, p):  # out_feats:64
        super(CRD, self).__init__()
        self.conv = GraphConv(in_feats, n_hid)
        self.p = p
        # Attention layer
        self.attention = MultiHead_SelfAttention2(n_hid, n_hid, 4,
                                                  0.1)  # def __init__(self, input_size, hidden_size, num_attention_heads, hidden_dropout_prob):
        # self.attention = GroupQueryAttention(out_feats, out_feats, 4, 0.1, 1)

    def reset_parameters(self):  # Parameter initialization
        # self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)

    def forward(self, g, x):  # x: torch.Size([5000, 216])
        x = self.conv(g, x)  # torch.Size([5000, 64])
        x = self.attention(x)  # 5000 * 64
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        return x

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]


class CLS(torch.nn.Module):
    def __init__(self, n_hid, n_class):
        super(CLS, self).__init__()
        self.conv = GraphConv(n_hid, n_class)
        # Attention layer
        self.attention = MultiHead_SelfAttention2(n_class, n_class, 1, 0.1)

    def reset_parameters(self):
        # self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)

    def forward(self, g, x):
        x = self.conv(g, x)
        # Attention layer
        x = self.attention(x)  # 5000 *27
        x = F.softmax(x, dim=1)
        return x

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]


class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size=64):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        temp = (beta * z).sum(1)
        return temp


class ImprovGCN(torch.nn.Module):
    def __init__(self, in_feat, n_hid, n_class, drop_out, dim_llms, hyper_adj_matrix):  # 216 n_hid:64
        super(ImprovGCN, self).__init__()
        self.crd = CRD(in_feat, n_hid, drop_out)
        self.cls = CLS(n_hid, n_class)  # n_out 27
        self.hyper_adj_matrix = hyper_adj_matrix
        self.dim_llms = dim_llms
        self.T_R = nn.Parameter(torch.FloatTensor(n_class, in_feat))
        self.soft = nn.Softmax(dim=1)
        self.attention = Attention(in_feat)
        torch.nn.init.xavier_uniform_(self.T_R)
        if self.dim_llms != 0:
            # self.mlp = nn.Sequential(
            #     nn.Linear(dim_llms+in_feat, 4096),
            #     nn.ReLU(),
            #     nn.Dropout(0.1),
            #     nn.Linear(4096, in_feat)
            # )
            self.linear = nn.Linear(dim_llms, in_feat)  # 将 N 的维度从 3000 转换为 100

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()
        if self.dim_llms != 0:
            # for layer in self.mlp:
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            self.linear.reset_parameters()

    def constraint(self):
        w = self.T_R.data.clamp(0, 1)
        col_sums = w.sum(dim=0) + 1e-10
        w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()
        self.T_R.data = w

    def forward(self, g, x, xl, llms_name, ):  # x:torch.Size([5000, 216])
        if xl.shape[1] != 0:
            if llms_name == "llama31_70b_instruct" or llms_name == "llama31_8b_instruct" or llms_name == 'qwen2_72b_instruct':
                xl = xl * 0.0001

            N_transformed = self.linear(xl)  # 变换 N 0.0023,  0.0017, -0.0081,  ...,  0.0023,  0.0110,  0.0057],
            # The results of GCN and LLMS were combined to train 46 communities and 29 attributes
            temp = torch.stack([x, N_transformed], dim=1) # 755 2 29
            x = self.attention(temp)

            # concat_x = torch.cat((x, xl), dim=1)
            # x = self.mlp(concat_x)

        x = self.crd(g, x)
        x = self.cls(g, x)  # torch.Size([5000, 27])  # [0.0079, 0.0637, 0.0984,  ..., 0.0488, 0.0170, 0.0586],

        hyper_out1 = self.hyper_adj_matrix @ x @ self.T_R  # reconstruction (attribute)
        hyper_out = self.soft(hyper_out1)
        return x, hyper_out

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]
