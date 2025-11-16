import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from models.gnn_modules import GNN_node, GNN_node_Virtualnode

from torch.distributions import Normal, Independent
from torch.nn.functional import softplus

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        return x

class InfoAlignModule(nn.Module):
    def __init__(self, configs):
        super(InfoAlignModule, self).__init__()
        self.graph_encoder = GNN_node_Virtualnode(
            num_layer=configs["model"]["graph_encoder"]["num_layers"],
            emb_dim=configs["model"]["graph_encoder"]["emb_dim"],
            JK=configs["model"]["graph_encoder"]["JK"],
            drop_ratio=configs["model"]["graph_encoder"]["drop_ratio"],
            residual=configs["model"]["graph_encoder"]["residual"],
            gnn_name=configs["model"]["graph_encoder"]["gnn_name"],
            norm_layer=configs["model"]["graph_encoder"]["norm_layer"],
        )

        graph_pooling = configs["model"]["graph_encoder"]["graph_pooling"]
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Invalid graph pooling method: {graph_pooling}")  

        self.dist_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(configs["model"]["graph_encoder"]["emb_dim"], 2 * configs["model"]["graph_encoder"]["emb_dim"], bias=True)
        )

        self.task_decoder = MLP(configs["model"]["graph_encoder"]["emb_dim"], hidden_features=4 * configs["model"]["graph_encoder"]["emb_dim"], out_features=configs["dataset"]["num_tasks"])
    
    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        mu, _ = self.dist_net(h_graph).chunk(2, dim=1)
        task_out = self.task_decoder(mu)
        return task_out
    
    def load_pretrained_graph_encoder(self, model_path):
        saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        graph_encoder_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('graph_encoder.')}
        graph_encoder_state_dict = {key.replace('graph_encoder.', ''): value for key, value in graph_encoder_state_dict.items()}
        self.graph_encoder.load_state_dict(graph_encoder_state_dict)
        # Load dist_net state dictionary
        dist_net_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('dist_net.')}
        dist_net_state_dict = {key.replace('dist_net.', ''): value for key, value in dist_net_state_dict.items()}
        self.dist_net.load_state_dict(dist_net_state_dict)
        self.freeze_graph_encoder()
    
    def freeze_graph_encoder(self):
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        for param in self.dist_net.parameters():
            param.requires_grad = False