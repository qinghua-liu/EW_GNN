import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing
# from torch_geometric.utils import softmax

# 区分incoming和outgoing
# incoming和outgoingfen别使用不同的GCN


class MLPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.drop = drop
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = nn.BatchNorm1d(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.relu:
            nn.init.kaiming_uniform_(self.lin.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_uniform_(self.lin.weight)
        else:
            nn.init.xavier_uniform_(self.lin.weight)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        x = self.lin(x)
        if self.batch and x.size()[0] > 1:
            x = self.BN(x)
        if self.drop:
            x = self.drop(x)
        x = self.activation(x)
        return x


class Initialization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Initialization, self).__init__()
        self.embedding = nn.Embedding(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        indices = torch.argmax(x, dim=1)
        output = self.embedding(indices)
        return output


class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GNN, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight):
        x = torch.matmul(x, self.weight)
        out = self.propagate(edge_index,
                             x=x,
                             edge_weight=edge_weight,
                             size=None)
        out += self.bias
        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class EdgeWeightedGCN(nn.Module):
    def __init__(self, node_out_channels, edge_attr_channels):
        super(EdgeWeightedGCN, self).__init__()
        self.embedding_in = GNN(node_out_channels,
                                node_out_channels,
                                flow="source_to_target")
        self.embedding_out = GNN(node_out_channels,
                                 node_out_channels,
                                 flow="target_to_source")
        self.W_in_0 = \
            nn.Parameter(torch.Tensor(2*node_out_channels+edge_attr_channels,
                                      edge_attr_channels))
        self.W_in_1 = nn.Parameter(torch.Tensor(edge_attr_channels, 1))
        self.W_out_0 = \
            nn.Parameter(torch.Tensor(2*node_out_channels+edge_attr_channels,
                                      edge_attr_channels))
        self.W_out_1 = nn.Parameter(torch.Tensor(edge_attr_channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_in_0)
        nn.init.xavier_uniform_(self.W_in_1)
        nn.init.xavier_uniform_(self.W_out_0)
        nn.init.xavier_uniform_(self.W_out_1)

    def forward(self, x, edge_attr_x, edge_index):
        # edge_index: incoming
        row, col = edge_index[0], edge_index[1]
        edge_attr_x_in, edge_attr_x_out = torch.chunk(edge_attr_x,
                                                      chunks=2,
                                                      dim=0)
        edge_info_in = torch.cat([x[row], edge_attr_x_in, x[col]], dim=1)
        edge_info_out = torch.cat([x[col], edge_attr_x_out, x[row]], dim=1)
        edge_attr_x_in = torch.matmul(edge_info_in, self.W_in_0)
        edge_attr_x_out = torch.matmul(edge_info_out, self.W_out_0)
        edge_attr_x = torch.cat([edge_attr_x_in, edge_attr_x_out], dim=0)
        edge_weight_in = torch.sigmoid(
            torch.matmul(edge_attr_x_in, self.W_in_1)).view(-1)
        edge_weight_out = torch.sigmoid(
            torch.matmul(edge_attr_x_out, self.W_out_1)).view(-1)
        x_in = self.embedding_in(x, edge_index, edge_weight=edge_weight_in)
        x_out = self.embedding_out(x, edge_index, edge_weight=edge_weight_out)
        message = x_in + x_out
        x = x + message
        return x, edge_attr_x


class DAGEmbedding(nn.Module):
    def __init__(self, node_out_channels, edge_attr_channels, layers):
        super(DAGEmbedding, self).__init__()
        self.K = layers
        self.emb = nn.ModuleList([
            EdgeWeightedGCN(node_out_channels, edge_attr_channels)
            for _ in range(layers)
        ])

    def forward(self, x, edge_attr_x, edge_index):
        for i in range(self.K):
            x, edge_attr_x = self.emb[i](x, edge_attr_x, edge_index)
        return x


class DAGPooling(nn.Module):
    def __init__(self):
        super(DAGPooling, self).__init__()

    def forward(self, x, batch_index):
        output = global_mean_pool(x, batch_index)
        return output


class Classifier(nn.Module):
    def __init__(self, node_out_channels):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            MLPBlock(2 * node_out_channels, node_out_channels),
            MLPBlock(node_out_channels, 2, activation=lambda x: x))

    def forward(self, conj_batch, prem_batch):
        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)
        return pred_y


class PremiseSelectionModel(nn.Module):
    def __init__(self, node_in_channels, node_out_channels,
                 edge_attr_in_channels, edge_attr_out_channels, layers):
        super(PremiseSelectionModel, self).__init__()
        self.initial_node = Initialization(node_in_channels, node_out_channels)
        self.initial_edge_attr = Initialization(edge_attr_in_channels,
                                                edge_attr_out_channels)
        self.dag_emb = DAGEmbedding(node_out_channels, edge_attr_out_channels,
                                    layers)
        self.pooling = DAGPooling()
        self.classifier = Classifier(node_out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.corrects = None

    def forward(self, batch):
        h_s = self.initial_node(batch.x_s)
        h_t = self.initial_node(batch.x_t)
        h_e_s = self.initial_edge_attr(batch.edge_attr_s)
        h_e_t = self.initial_edge_attr(batch.edge_attr_t)
        h_s = self.dag_emb(h_s, h_e_s, batch.edge_index_s)
        h_t = self.dag_emb(h_t, h_e_t, batch.edge_index_t)
        h_g_s = self.pooling(h_s, batch.x_s_batch)
        h_g_t = self.pooling(h_t, batch.x_t_batch)
        pred_y = self.classifier(h_g_s, h_g_t)
        pred_label = torch.max(pred_y, dim=1)[1]
        self.corrects = (pred_label == batch.y).sum().cpu().item()
        loss = self.criterion(pred_y, batch.y)
        return loss
