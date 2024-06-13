import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCNModel, self).__init__()

        self.norm1 = nn.InstanceNorm1d(num_features)
        self.norm2 = nn.InstanceNorm1d(num_features // 2)
        self.norm3 = nn.InstanceNorm1d(num_features // 2)

        self.embed1 = nn.Linear(num_features, num_features // 2)
        self.embed2 = nn.Linear(num_features//2, num_features//2)
        self.embed3 = nn.Linear(num_features//2, num_features)


        self.norm4 = nn.InstanceNorm1d(227)
        self.norm5 = nn.InstanceNorm1d(227//2)
        self.norm6 = nn.InstanceNorm1d(227//2)
        self.embed4 = nn.Linear(227, 227 // 2)
        self.embed5 = nn.Linear(227//2, 227//2)
        self.embed6 = nn.Linear(227//2, 227)

        self.norm7 = nn.InstanceNorm1d(num_features+227)

        self.conv1 = GCNConv(num_features+227, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # self.conv1 = GCNConv(227, 227)
        # self.conv2 = GCNConv(227, 227)
        # self.conv3 = GCNConv(227, 227)
        
        self.fc1 = nn.Linear(hidden_channels, hidden_channels//2)
        self.fc2 = nn.Linear(hidden_channels//2, 1)
        self.fc0 = nn.Linear(num_features+227, num_features+227)

        # self.fc1 = nn.Linear(227, 227//2)
        # self.fc2 = nn.Linear(227//2, 1)
        
    def forward(self, x, edge_index):
        x1 = x[:,:3072]
        x2 = x[:,3072:]

        x1 = self.norm1(x1.permute(1, 0))
        x1 = self.embed1(x1.permute(1, 0))
        x1 = F.relu(x1)
        
        x1 = self.norm2(x1.permute(1, 0))
        x1 = self.embed2(x1.permute(1, 0))
        x1 = F.relu(x1)

        x1 = self.norm3(x1.permute(1, 0))
        x1 = self.embed3(x1.permute(1, 0))
        x1 = F.relu(x1)
        
        x2 = self.norm4(x2.permute(1, 0))
        x2 = self.embed4(x2.permute(1, 0))
        x2 = F.relu(x2)

        x2 = self.norm5(x2.permute(1, 0))
        x2 = self.embed5(x2.permute(1, 0))
        x2 = F.relu(x2)

        x2 = self.norm6(x2.permute(1, 0))
        x2 = self.embed6(x2.permute(1, 0))
        x2 = F.relu(x2)

        x = torch.concat((x1, x2), axis=1)
        x = self.norm7(x.permute(1,0))
        x = x.permute(1,0)
        # x = x2
        
        # 新增
        # x = self.fc0(x)
        # x = F.relu(x)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        # x = F.relu(x)
        
        # x = self.fc3(x)

        return F.sigmoid(x)