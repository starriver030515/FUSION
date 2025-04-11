import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
import math
import numpy as np


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.k_proj = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.v_proj = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, q_dim, bias=attention_bias)

    def forward(self, queries, vision_latents, attention_mask=None):
        bsz, q_len, _ = queries.size()
        bsz, v_len, _ = vision_latents.size()

        query_states = self.q_proj(queries)
        key_states = self.k_proj(vision_latents)
        value_states = self.v_proj(vision_latents)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, v_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__() 
        self.linear_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))
    

class VisionCrossAttentionLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim=1024, num_heads=16):
        super(VisionCrossAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim

        self.proj_in = nn.Linear(q_dim * 2, hidden_dim, bias=False)  
        self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(q_dim, kv_dim, hidden_dim, num_heads)

        self.proj_context = nn.Linear(kv_dim, hidden_dim, bias=False) 

    def forward(self, hidden_states, global_ctx, vision_pos, instruct_pos, queries, image_tokens, query_len):
        B, _, D = hidden_states.size()
        _, G, _ = global_ctx.size()
        B, L = instruct_pos.size()
        V = L * query_len

        residual = image_tokens 

        concat_queries = torch.cat([queries, image_tokens], dim=-1)  # (B, V, 2D)
        projected_queries = self.proj_in(concat_queries)  # (B, V, hidden_dim)

        H_q = W_q = int(math.sqrt(query_len))
        H_gc = W_gc = int(math.sqrt(G))
        H_win = H_gc // H_q
        W_win = W_gc // W_q

        ctx_feat = global_ctx.view(B, H_gc, W_gc, D).permute(0, 3, 1, 2)  # (B, D, H_gc, W_gc)
        windows = F.unfold(ctx_feat, kernel_size=(H_win, W_win), stride=(H_win, W_win))  # (B, D * H_win * W_win, query_len)
        windows = windows.view(B, D, H_win * W_win, -1).permute(0, 3, 2, 1)  # (B, query_len, H_win * W_win, D)
        windows = windows.unsqueeze(1).expand(-1, L, -1, -1, -1).contiguous().view(B, V, H_win * W_win, D)  # (B, V, H_win * W_win, D)
        windows = windows.view(B * V, H_win * W_win, D)  # (B*V, H_win*W_win, D)
        windows = self.proj_context(windows)  # (B*V, H_win * W_win, hidden_dim)

        queries_proj = projected_queries.view(B * V, 1, D)  # (B*V, 1, hidden_dim)

        attn_output = self.cross_attn(queries_proj, windows)  # (B*V, 1, hidden_dim)
        attn_output = attn_output.view(B, V, -1)  # (B, V, q_dim)
        attn_output = self.norm(attn_output + projected_queries)  # (B, V, q_dim)
        out = self.proj_out(attn_output)  # (B, V, q_dim)
        out = out + residual
        return out  # Return the modified hidden_states
    
class VisionTokenSampler(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_size=1024, num_of_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([VisionCrossAttentionLayer(q_dim, kv_dim, hidden_size) for _ in range(num_of_layers)])

    def forward(self, hidden_states, global_ctx, vision_pos, instruct_pos, queries, image_tokens, query_len):
        for layer in self.layers:
            queries = layer(hidden_states, global_ctx, vision_pos, instruct_pos, queries, image_tokens, query_len)

        return queries