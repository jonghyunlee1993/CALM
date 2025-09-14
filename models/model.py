import math
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_to_device(v, device) for v in x)
    return x

# -----------------------------------------------------------------------------
# Align Layer
# -----------------------------------------------------------------------------
class AlignLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layers(x)

# -----------------------------------------------------------------------------
# Image Encoding Block (batch_first=True)
# -----------------------------------------------------------------------------
class ImageEncodingBlock(nn.Module):
    """
    x: [B, S, E]
    """
    def __init__(self, hidden_dim, dropout_rate=0.1, num_heads=4, **kwargs):
        super().__init__()
        self.alignment = AlignLayer(hidden_dim, hidden_dim, dropout_rate)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.post_layers = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity()
        )

    def forward(self, x):
        x = self.alignment(x)                 # [B,S,E]
        x, A = self.attention_layer(x, x, x)  # [B,S,E], [B, heads, S, S] (avg 설정에 따름)
        x = self.post_layers(x)
        return x, A

# -----------------------------------------------------------------------------
# Text Encoder (MUSK/BERT)
# -----------------------------------------------------------------------------
from musk import modeling as musk_modeling
from timm.models import create_model
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, text_encoder_type: str = "musk", out_dim: int = 512, lora_cfg: dict = None):
        super().__init__()
        self.mode = text_encoder_type
        self.out_dim = out_dim

        if self.mode == "musk":
            # MUSK(beIT3) backbone
            model_config = "musk_large_patch16_384"
            model = create_model(model_config).eval()
            from musk import utils as musk_utils
            musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
            in_dim = 1024
            for p in model.parameters():
                p.requires_grad = False
            self.model = model
            self._impl = "musk"

        elif self.mode == "bert":
            model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
            model = BertModel.from_pretrained(model_name)
            in_dim = 768
            for p in model.parameters():
                p.requires_grad = False
            # optional LoRA
            if lora_cfg is not None:
                from peft import get_peft_model, LoraConfig
                lora = LoraConfig(
                    r=lora_cfg.get("r", 8),
                    lora_alpha=lora_cfg.get("alpha", 16),
                    target_modules=lora_cfg.get("target_modules", ["query", "key", "value"]),
                    lora_dropout=lora_cfg.get("dropout", 0.1),
                    bias="none",
                )
                model = get_peft_model(model, lora)
            self.model = model
            self._impl = "bert"

        else:
            raise NotImplementedError(f"Unknown text_encoder_type: {self.mode}")

        self.text_alignment = AlignLayer(in_dim=in_dim, out_dim=out_dim)

    @torch.no_grad()
    def get_text_embedding(self, input_texts):
        """
        Returns hidden states (before alignment).
        MUSK: input_texts is tensor of token ids or pre-shaped tokens
        BERT: input_texts is BatchEncoding(dict) with input_ids/attention_mask...
        """
        if self._impl == "musk":
            x = input_texts
            if isinstance(x, torch.Tensor):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                elif x.dim() == 3 and x.size(0) == 1:
                    pass
            else:
                x = torch.tensor(x, device=next(self.model.parameters()).device)
            out = self.model.beit3(textual_tokens=x)
            hidden = out["encoder_out"]  # [B, S, 1024]
            return hidden

        elif self._impl == "bert":
            outputs = self.model(**input_texts, output_hidden_states=False)
            hidden = outputs.last_hidden_state  # [B, S, 768]
            return hidden

    def forward(self, input_texts):
        device = next(self.model.parameters()).device
        if self._impl == "bert":
            input_texts = _to_device(input_texts, device)
        else:
            if isinstance(input_texts, torch.Tensor):
                input_texts = input_texts.to(device)
            else:
                input_texts = torch.tensor(input_texts, device=device)
        hidden = self.get_text_embedding(input_texts)            # [B,S,in_dim]
        return self.text_alignment(hidden)                       # [B,S,out_dim]

# -----------------------------------------------------------------------------
# Attention Pooling (with pad mask)
# -----------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None):
        """
        x: [B,S,E]
        pad_mask: [B,S], True=PAD(무시)
        """
        attn_scores = self.attention(x).squeeze(-1)       # [B,S]
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(pad_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B,S]
        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # [B,E]
        return pooled

# -----------------------------------------------------------------------------
# Standard Transformer Block (Pre-LN, batch_first=True)
# -----------------------------------------------------------------------------
class StandardAttentionBlockBF(nn.Module):
    """
    Pre-LN Transformer 블록 (Self/Cross 겸용, batch_first=True)
    입력/출력: [B, S, E]
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 ffn_ratio: float = 4.0, bias: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, bias=bias, batch_first=True
        )
        self.ln_q  = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)

        ffn_hidden = int(hidden_dim * ffn_ratio)
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self,
                query: torch.Tensor,           # [B,S_q,E]
                key: Optional[torch.Tensor] = None,   # [B,S_k,E]
                value: Optional[torch.Tensor] = None, # [B,S_k,E]
                *,
                key_padding_mask: Optional[torch.Tensor] = None,  # [B,S_k], True=PAD
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True):
        B, S_q, E = query.shape
        q = self.ln_q(query)
        if key is None:
            k = v = q
        else:
            k = self.ln_kv(key)
            v = self.ln_kv(value if value is not None else key)

        attn_out, attn_w = self.attn(
            q, k, v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=False  # [B, heads, S_q, S_k]
        )

        x = query + attn_out                 # Residual 1
        y = self.ffn(self.ln_ffn(x))
        out = x + y                          # Residual 2

        if need_weights:
            attn_w = attn_w.mean(dim=1)     # [B, S_q, S_k]
            return out, attn_w
        return out, None

# -----------------------------------------------------------------------------
# Self / Cross Attention Blocks
# -----------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, num_heads=8, **_):
        super().__init__()
        self.block = StandardAttentionBlockBF(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate
        )

    def forward(self, feat, key_padding_mask=None):
        out, A = self.block(feat, key_padding_mask=key_padding_mask, need_weights=True)
        return out, A

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, num_heads=8, **_):
        super().__init__()
        self.block_i2t = StandardAttentionBlockBF(hidden_dim, num_heads, dropout_rate)
        self.block_t2i = StandardAttentionBlockBF(hidden_dim, num_heads, dropout_rate)

    def forward(self, image_feat, text_feat,
                text_pad_mask: Optional[torch.Tensor] = None,
                image_pad_mask: Optional[torch.Tensor] = None):
        # image_to_text: Q=image, K/V=text
        i2t_feat, i2t_A = self.block_i2t(
            query=image_feat, key=text_feat, value=text_feat,
            key_padding_mask=text_pad_mask, need_weights=True
        )
        # text_to_image: Q=text, K/V=image
        t2i_feat, t2i_A = self.block_t2i(
            query=text_feat, key=image_feat, value=image_feat,
            key_padding_mask=image_pad_mask, need_weights=True
        )
        return i2t_feat, t2i_feat, i2t_A, t2i_A

# -----------------------------------------------------------------------------
# Feature Combination (symmetric residuals + pooling with mask)
# -----------------------------------------------------------------------------
class FeatureCombination(nn.Module):
    def __init__(self, feature_dim, dropout_rate=0.1, num_heads=4, num_layers=2,
                 hidden_dim=512, text_encoder_type="musk",
                 use_text_cls: bool = True):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.num_layers = num_layers
        self.use_text_cls = use_text_cls

        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim=feature_dim, dropout_rate=dropout_rate, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.attention_pooling = AttentionPooling(hidden_dim=hidden_dim)

        # small combine FFN (Pre-LN 스타일에 맞춰 LN 먼저)
        self.combine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, image_feat, text_feat,
                text_pad_mask: Optional[torch.Tensor] = None,
                image_pad_mask: Optional[torch.Tensor] = None,
                text_input_ids: Optional[torch.Tensor] = None,
                tokenizer=None):
        """
        image_feat: [B,N,E], text_feat: [B,T,E]
        *_pad_mask: True=PAD
        """
        # Cross layers
        orig_image_feat, orig_text_feat = image_feat, text_feat
        for i in range(self.num_layers):
            image_to_text_feat, text_to_image_feat, image_to_text_A, text_to_image_A = \
                self.layers[i](image_feat, text_feat, text_pad_mask=text_pad_mask, image_pad_mask=image_pad_mask)
            image_feat, text_feat = image_to_text_feat, text_to_image_feat

        # Symmetric residuals
        image_feat = image_feat + orig_image_feat
        text_feat  = text_feat  + orig_text_feat

        # Pooling
        image_CLS_token = self.attention_pooling(image_feat, pad_mask=image_pad_mask)  # [B,E]

        if self.use_text_cls:
            # robust CLS pick (fallback to first token)
            if (text_input_ids is not None) and (tokenizer is not None) and (getattr(tokenizer, 'cls_token_id', None) is not None):
                cls_id = tokenizer.cls_token_id
                cls_pos = (text_input_ids == cls_id).int().argmax(dim=1)  # [B]
                b_idx = torch.arange(text_feat.size(0), device=text_feat.device)
                text_CLS_token = text_feat[b_idx, cls_pos]               # [B,E]
            else:
                text_CLS_token = text_feat[:, 0]                         # fallback
        else:
            text_CLS_token = self.attention_pooling(text_feat, pad_mask=text_pad_mask)

        combined = self.combine(image_CLS_token + text_CLS_token)        # [B,E]
        return combined, image_CLS_token, text_CLS_token, image_to_text_A, text_to_image_A

# -----------------------------------------------------------------------------
# MM Encoder (head + forward)
# -----------------------------------------------------------------------------
class MMEncoder(nn.Module):
    def __init__(self, image_encoder, text_encoder,
                 hidden_dim=512, n_classes=4, text_encoder_type="musk",
                 use_text_cls: bool = True):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder
        self.text_encoder_type = text_encoder_type

        self.feature_combination = FeatureCombination(
            feature_dim=hidden_dim, dropout_rate=0.1, num_heads=4, num_layers=2,
            hidden_dim=hidden_dim, text_encoder_type=self.text_encoder_type,
            use_text_cls=use_text_cls
        )

        self.fc_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self,
                image_inputs,                     # encoder-specific
                text_inputs,                      # MUSK: ids tensor / BERT: BatchEncoding
                diagnostic_description=None,      # optional second text stream
                *,
                text_input_ids: Optional[torch.Tensor] = None,   # [B,T] (for CLS pick)
                text_attention_mask: Optional[torch.Tensor] = None,  # [B,T] (True=1 valid)
                image_pad_mask: Optional[torch.Tensor] = None,   # [B,N] (True=PAD)
                tokenizer=None):
        """
        Returns:
            out, mixed_feat, diag_desc(emb, detached), image_CLS, text_CLS, i2t_A, t2i_A
        """
        # Encode
        image_feat = self.image_encoder(image_inputs)       # [B,N,E]
        text_feat  = self.text_encoder(text_inputs)         # [B,T,E]

        # Optional diagnostic description embedding (pooled CLS)
        diag_desc_emb = None
        if diagnostic_description is not None:
            dd = self.text_encoder(diagnostic_description)  # [B,T,E]
            diag_desc_emb = dd[:, 0].detach()               # simple CLS (detached)

        # Masks
        text_pad_mask = None
        if text_attention_mask is not None:
            # attention_mask: 1=valid, 0=pad  → pad_mask=True on pads
            text_pad_mask = (text_attention_mask == 0)

        # Combine
        mixed_feat, image_CLS, text_CLS, i2t_A, t2i_A = self.feature_combination(
            image_feat, text_feat,
            text_pad_mask=text_pad_mask,
            image_pad_mask=image_pad_mask,
            text_input_ids=text_input_ids,
            tokenizer=tokenizer
        )

        out = self.fc_head(mixed_feat)
        return out, mixed_feat, diag_desc_emb, image_CLS, text_CLS, i2t_A, t2i_A