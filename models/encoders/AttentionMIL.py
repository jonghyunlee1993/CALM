import torch
import torch.nn as nn
import torch.nn.functional as F

# A Modified Implementation of Deep Attention MIL & PORPOISE (https://github.com/mahmoodlab/PORPOISE)
class GatedAttentionNet(nn.Module):
    """Implements Gated Attention mechanism."""
    def __init__(self, L=1024, D=512, dropout=False, n_classes=1):
        super().__init__()
        dropout_rate = 0.25

        # Initialize attention networks
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            *(nn.Dropout(dropout_rate),) if dropout else ()
        )
        
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            *(nn.Dropout(dropout_rate),) if dropout else ()
        )

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(0)
        
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b
        A = self.attention_c(A)  # N x n_classes

        return A, x

class GatedAttentionMIL(nn.Module):
    """Implements MIL using GatedAttentionNet."""
    def __init__(self, hidden_dims=[1024, 512, 256], dropout=False, **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims

        # Construct the attention network and classifier
        attention_net_layers = [
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU()
        ]

        if dropout:
            attention_net_layers.append(nn.Dropout(0.25))

        attention_net_layers.append(
            GatedAttentionNet(L=self.hidden_dims[1], D=self.hidden_dims[2], dropout=dropout, n_classes=1)
        )

        self.attention_net = nn.Sequential(*attention_net_layers)
        
    def forward(self, h):
        A, h = self.attention_net(h)  
        # A = torch.transpose(A, 1, 0) 
        # A = F.softmax(A, dim=1)
        # M = torch.mm(A, h)  # Weighted sum based on attention
        
        # results_dict = {
        #     "MIL_attention": A,
        #     "MIL_out": M,
        #     "hidden_state": h 
        # }
        
        # return results_dict

        return h.unsqueeze(0)

if __name__ == "__main__":
    # Example usage
    x = torch.rand((776, 1024))  # 10 instances with 1024 features each
    model = GatedAttentionMIL()
    
    print(model(x))
    # print(model(x)['MIL_attention'].shape)
    # print(model(x)['MIL_out'].shape)
    # print(model(x)['hidden_state'].shape)