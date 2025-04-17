from models.encoders.AttentionMIL import GatedAttentionMIL
from models.layer import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class AlignLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()

        self.layers = [
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity()
        ]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ImageEncodingBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, num_heads=4, **kwargs):
        super().__init__()

        self.alignment = AlignLayer(hidden_dim, hidden_dim, dropout_rate)
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)
        
        self.post_layers = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity()
        )
        self.post_layers = nn.Sequential(*self.post_layers)
    
    def forward(self, x):
        x = self.alignment(x) # B S D        
        x, A = self.attention_layer(x, x, x)
        x = self.post_layers(x)
        
        return x, A


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, num_heads=4, **kwargs):
        super().__init__()
        
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)
        self.post_layers = [
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ]
        
        if dropout_rate != 0:
            self.post_layers.append(nn.Dropout(dropout_rate))
            
        self.post_layers = nn.Sequential(*self.post_layers)

    def forward(self, query, key, value):
        x, A = self.attention_layer(query, key, value)
        x = self.post_layers(x)

        return x, A

class TextEncoder(nn.Module):
    def __init__(self, freeze_layer_index=12):
        super().__init__()

        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        self.__freeze_weights(freeze_layer_index=freeze_layer_index)
        
        self.text_alignment = AlignLayer(in_dim=768, out_dim=512)
    
    def __freeze_weights(self, freeze_layer_index=12):
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        for layer in self.model.encoder.layer[:freeze_layer_index]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, **inputs):
        outputs = self.model(**inputs, output_hidden_states=False)
        hidden_state = outputs.last_hidden_state

        return self.text_alignment(hidden_state)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_sum = torch.sum(attn_weights * x, dim=1)
        
        return weighted_sum


class MMEncoder(nn.Module):
    def __init__(self, image_encoder, text_encoder, hidden_dim=512, n_classes=4, is_CLS=False, temperature=1):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
               
        self.feature_combination = FeatureCombination(feature_dim=512, dropout_rate=0.1, num_heads=4, num_layers=2, hidden_dim=512, is_CLS=is_CLS, temperature=temperature)
        
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )
    
    def forward(self, image_inputs, text_inputs, diagnostic_description):
        image_feat = self.image_encoder(image_inputs)
        try:
            text_feat = self.text_encoder(text_inputs)
        except:
            text_feat = self.text_encoder(**text_inputs)
        mixed_feat, image_CLS_token, text_CLS_token, image_to_text_A, text_to_image_A = self.feature_combination(image_feat, text_feat)
        out = self.fc_head(mixed_feat)

        diagnostic_description = self.text_encoder(**diagnostic_description)

        return out, mixed_feat, diagnostic_description.detach(), image_CLS_token, text_CLS_token, image_to_text_A, text_to_image_A


class FeatureCombination(nn.Module):
    def __init__(self, feature_dim, dropout_rate=0.1, num_heads=4, num_layers=2, hidden_dim=512, is_CLS=False, temperature=0.5):
        super(FeatureCombination, self).__init__()
        self.is_CLS = is_CLS
        if not self.is_CLS:
            self.attention_pooling = AttentionPooling(hidden_dim=hidden_dim)
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([CrossAttentionBlock(hidden_dim=feature_dim, dropout_rate=dropout_rate, num_heads=num_heads, temperature=temperature) for _ in range(num_layers)])
        
        self.post_layers = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity()
        )
        
    def forward(self, image_feat, text_feat):
        orig_image_feat, orig_text_feat = image_feat, text_feat
        
        for i in range(self.num_layers):
            image_to_text_feat, text_to_image_feat, image_to_text_A, text_to_image_A = self.layers[i](image_feat, text_feat)
            image_feat, text_feat = image_to_text_feat, text_to_image_feat
            
        image_feat += orig_image_feat
        text_feat += self.post_layers(orig_text_feat)
        
        if self.is_CLS:
            image_CLS_token = image_feat[:, 0] 
        else:
            image_CLS_token = self.attention_pooling(image_feat)
            
        text_CLS_token = text_feat[:, 0]
        
        combined = image_CLS_token + text_CLS_token
        combined = self.post_layers(combined)
        
        return combined, image_CLS_token, text_CLS_token, image_to_text_A, text_to_image_A


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, num_heads=4, temperature=0.5, **kwargs):
        super().__init__()

        self.self_attention = AttentionBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate, num_heads=num_heads, temperature=temperature)
        
    def transpose_tensor(self, x):
        return x.transpose(0, 1)
    
    def forward(self, feat):
        feat = self.transpose_tensor(feat)
        feat, _ = self.self_attention(feat, feat, feat)
        feat = self.transpose_tensor(feat)        
        
        return feat


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, num_heads=4, temperature=0.5, **kwargs):
        super().__init__()

        self.cross_attention_image_to_text = AttentionBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate, num_heads=num_heads, temperature=temperature)
        self.cross_attention_text_to_image = AttentionBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate, num_heads=num_heads, temperature=temperature)
        
    def transpose_tensor(self, x):
        return x.transpose(0, 1)
    
    def forward(self, image_feat, text_feat):
        image_feat = self.transpose_tensor(image_feat)
        text_feat = self.transpose_tensor(text_feat)
        
        image_to_text_feat, image_to_text_A = self.cross_attention_image_to_text(image_feat, text_feat, text_feat)
        text_to_image_feat, text_to_image_A = self.cross_attention_text_to_image(text_feat, image_feat, image_feat)

        image_to_text_feat = self.transpose_tensor(image_to_text_feat)
        text_to_image_feat = self.transpose_tensor(text_to_image_feat)
        
        return image_to_text_feat, text_to_image_feat, image_to_text_A, text_to_image_A


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.1, temperature=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, query, key, value):
        # query/key/value: [seq_len, batch, hidden_dim]
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # reshape for multihead: [seq_len, batch, num_heads, head_dim]
        def reshape(x):
            return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        Q = reshape(Q)
        K = reshape(K)
        V = reshape(V)

        # attention score: [batch, num_heads, q_len, k_len]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, q_len, head_dim]

        # reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(query.size(1), query.size(0), self.hidden_dim)
        attn_output = attn_output.transpose(0, 1)

        out = self.out_proj(attn_output)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.norm(out)

        return out, attn_weights.mean(dim=1)  # return mean over heads