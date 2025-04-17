import torch
import torch.nn as nn
import torch.nn.functional as F


class KroneckerProduct(nn.Module):
    def __init__(self, gate1=True, gate2=True, dim1=512, dim2=512, scale_dim1=16, scale_dim2=16, mmhid=512, dropout_rate=0.25, **kwargs):
        super().__init__()
        self.gate1 = gate1
        self.gate2 = gate2
        # Original and scaled dimensions
        dim1_og, dim2_og = dim1, dim2
        dim1, dim2 = dim1 // scale_dim1, dim2 // scale_dim2

        # Define layers for the first input vector
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Linear(dim1_og + dim2_og, dim1)
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Define layers for the second input vector
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Linear(dim1_og + dim2_og, dim2)
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Define post-fusion and encoder layers
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256, mmhid), nn.ReLU())

    def forward(self, vec1, vec2):
        # Process the first vector
        h1 = self.linear_h1(vec1)
        o1 = self.process_vector(h1, vec1, vec2, self.gate1, self.linear_z1, self.linear_o1)

        # Process the second vector
        h2 = self.linear_h2(vec2)
        o2 = self.process_vector(h2, vec1, vec2, self.gate2, self.linear_z2, self.linear_o2)

        # Fusion and encoding
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=o1.device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=o2.device)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        out = self.encoder2(out)
        return out

    def process_vector(self, h, vec1, vec2, gate, linear_z, linear_o):
        if gate:
            z = linear_z(torch.cat((vec1, vec2), dim=1))
            return linear_o(F.sigmoid(z) * h)
        else:
            return linear_o(h)


class ConcatenateLayer(nn.Module):
    def __init__(self, dim1=512, dim2=512, scale_dim1=2, scale_dim2=2, dropout=True, dropout_rate=0.25, **kwargs):
        super().__init__()
                
        if scale_dim1 != 1:        
            self.proj_1 = nn.Linear(dim1, dim1 // scale_dim1, bias=False)
        else:
            self.proj_1 = nn.Identity()
        
        if scale_dim2 != 1:
            self.proj_2 = nn.Linear(dim2, dim2 // scale_dim2, bias=False)
        else:
            self.proj_2 = nn.Identity()
        
        if dropout:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()
        
    def forward(self, x, y):        
        x = self.proj_1(x)
        y = self.proj_2(y)
        
        xy = torch.cat((x, y), dim=1)

        return self.dropout(xy)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim_key, embed_dim_query, num_heads):
        # Q: Text / K: Image ; text reports generated from images
        
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim_key = embed_dim_key // num_heads
        self.head_dim_query = embed_dim_query // num_heads

        assert self.head_dim_key * num_heads == embed_dim_key, "embed_dim_key must be divisible by num_heads"
        assert self.head_dim_query * num_heads == embed_dim_query, "embed_dim_query must be divisible by num_heads"

        self.query_proj = nn.Linear(embed_dim_key, embed_dim_key)
        self.key_proj = nn.Linear(embed_dim_query, embed_dim_key)
        self.value_proj = nn.Linear(embed_dim_query, embed_dim_key)
        
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim_key, embed_dim_key)

    def forward(self, key_features, query_features):
        bsz = key_features.size(0)

        # Linear projections
        queries = self.query_proj(key_features)  # (b, seq_len_key, embed_dim_key)
        keys = self.key_proj(query_features)      # (b, seq_len_query, embed_dim_key)
        values = self.value_proj(query_features)  # (b, seq_len_query, embed_dim_key)

        # Split heads
        queries = queries.view(bsz, -1, self.num_heads, self.head_dim_key).transpose(1, 2)  # (b, num_heads, seq_len_key, head_dim_key)
        keys = keys.view(bsz, -1, self.num_heads, self.head_dim_key).transpose(1, 2)        # (b, num_heads, seq_len_query, head_dim_key)
        values = values.view(bsz, -1, self.num_heads, self.head_dim_key).transpose(1, 2)    # (b, num_heads, seq_len_query, head_dim_key)

        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim_key ** 0.5)  # (b, num_heads, seq_len_key, seq_len_query)
        attn_scores = nn.functional.softmax(attn_scores, dim=-1)  # (b, num_heads, seq_len_key, seq_len_query)
        attn_scores = self.attn_drop(attn_scores)

        # Compute attention output
        conkey = torch.matmul(attn_scores, values)  # (b, num_heads, seq_len_key, head_dim_key)

        # Concatenate heads
        conkey = conkey.transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim_key)  # (b, seq_len_key, embed_dim_key)

        # Final linear projection
        output = self.proj(conkey)  # (b, seq_len_key, embed_dim_key)

        return output, attn_scores 


class MLP(nn.Module):
    def __init__(self, 
                 hidden_dims=[3072, 1024, 512], 
                 dropout=[True, True], 
                 dropout_rate=[0.25, 0.25], 
                 activation=[True, True], 
                 **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        # Construct layers based on hidden dimensions
        for i in range(len(hidden_dims) - 1):
            self.add_layer_block(hidden_dims[i], hidden_dims[i + 1], dropout[i], dropout_rate[i], activation[i])
            
    def add_layer_block(self, input_dim, output_dim, dropout, dropout_rate, activation):
        """ Helper function to add a block of layers including Linear, ReLU, and optionally Dropout. """
        self.layers.append(nn.Linear(input_dim, output_dim))
        
        if activation:
            self.layers.append(nn.ReLU())
        
        if dropout:
            self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        """ Forward pass through the network, applying each layer in sequence. """
        for layer in self.layers:
            x = layer(x)
        return x


class SNN(nn.Module):
    def __init__(self, **kwargs):
        pass
    
    def forward(self, x):
        pass


if __name__ == "__main__":
    x = torch.rand((1, 512))
    y = torch.ran((1, 512))    
    model = ConcatenateLayer()

    print(model(x, y).shape)  # Should print torch.Size([10, 1024])