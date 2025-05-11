import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, hidden_dim=128,hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim),nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)
    

class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim=128, hidden_layers=3):
        super().__init__()
        self.input_layer = nn.Linear(3, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU()  
        ] * hidden_layers)
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x_res = self.input_layer(x)
        x_res = self.relu(x_res)
        for i in range(0, len(self.hidden_layers), 2): 
            residual = x_res  
            x_res = self.hidden_layers[i](x_res) 
            x_res = self.hidden_layers[i + 1](x_res)  
            x_res = x_res + residual  
        x_out = self.output_layer(x_res)  
        return x_out