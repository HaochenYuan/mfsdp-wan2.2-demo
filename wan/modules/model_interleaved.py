import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention
from .model import (
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    WanCrossAttention,
    Head,
)

__all__ = [
    'WanModelInterleaved',
    'WanFirstUnit', 'WanStaggeredUnit', 'WanLastUnit',
]


class WanAttnHalf(nn.Module):
    """Self-attention + Cross-attention half of a transformer layer."""

    def __init__(self, dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (WanLayerNorm(dim, eps, elementwise_affine=True)
                      if cross_attn_norm else nn.Identity())
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.modulation = nn.Parameter(torch.randn(1, 3, dim) / dim ** 0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_parts = (self.modulation.unsqueeze(0) + e[:, :, :3, :]).chunk(3, dim=2)

        y = self.self_attn(
            self.norm1(x).float() * (1 + e_parts[1].squeeze(2)) + e_parts[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_parts[2].squeeze(2)

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        return x


class WanFFNHalf(nn.Module):
    """FFN half of a transformer layer."""

    def __init__(self, dim, ffn_dim, eps=1e-6):
        super().__init__()
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 3, dim) / dim ** 0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_parts = (self.modulation.unsqueeze(0) + e[:, :, 3:, :]).chunk(3, dim=2)

        y = self.ffn(
            self.norm2(x).float() * (1 + e_parts[1].squeeze(2)) + e_parts[0].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_parts[2].squeeze(2)
        return x


class WanFirstUnit(nn.Module):
    """FSDP Unit #0: first layer's attention half."""

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.attn_half = WanAttnHalf(dim, num_heads, window_size,
                                     qk_norm, cross_attn_norm, eps)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        return self.attn_half(x, e, seq_lens, grid_sizes, freqs, context, context_lens)


class WanStaggeredUnit(nn.Module):
    """FSDP Unit #1..N-1: layer[i].ffn + layer[i+1].attn."""

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.ffn_half = WanFFNHalf(dim, ffn_dim, eps)
        self.attn_half = WanAttnHalf(dim, num_heads, window_size,
                                     qk_norm, cross_attn_norm, eps)

    def forward(self, x, e_ffn, e_attn, seq_lens, grid_sizes, freqs,
                context, context_lens):
        x = self.ffn_half(x, e_ffn)
        x = self.attn_half(x, e_attn, seq_lens, grid_sizes, freqs,
                           context, context_lens)
        return x


class WanLastUnit(nn.Module):
    """FSDP Unit #N: last layer's FFN half."""

    def __init__(self, dim, ffn_dim, eps=1e-6):
        super().__init__()
        self.ffn_half = WanFFNHalf(dim, ffn_dim, eps)

    def forward(self, x, e):
        return self.ffn_half(x, e)


class WanModelInterleaved(ModelMixin, ConfigMixin):
    """
    Wan diffusion backbone with interleaved FSDP unit layout.
    """

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings (identical to original WanModel)
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # interleaved units: N layers → N+1 FSDP units
        block_kwargs = dict(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
                            window_size=window_size, qk_norm=qk_norm,
                            cross_attn_norm=cross_attn_norm, eps=eps)

        self.units = nn.ModuleList()
        self.units.append(WanFirstUnit(**block_kwargs))
        for _ in range(num_layers - 1):
            self.units.append(WanStaggeredUnit(**block_kwargs))
        self.units.append(WanLastUnit(dim=dim, ffn_dim=ffn_dim, eps=eps))

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE freqs
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.init_weights()

    def forward(self, x, t, context, seq_len, y=None):
        if self.model_type == 'i2v':
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # forward through interleaved units
        # e0 shape: [B, seq_len, 6, dim]
        # Each layer needs its own e0 slice:
        #   attn_half uses e0[:, :, :3, :] (scale/shift/gate for self-attn)
        #   ffn_half uses e0[:, :, 3:, :] (scale/shift/gate for ffn)
        # But in this model the same e0 is shared across all layers
        # (same as original WanModel where all blocks receive the same e0).

        # Unit 0: FirstUnit = layer0.attn
        x = self.units[0](x, e0, seq_lens, grid_sizes, self.freqs,
                          context, context_lens)

        # Units 1..N-1: StaggeredUnit = layer[i-1].ffn + layer[i].attn
        for i in range(1, self.num_layers):
            x = self.units[i](x, e0, e0, seq_lens, grid_sizes, self.freqs,
                              context, context_lens)

        # Unit N: LastUnit = layer[N-1].ffn
        x = self.units[self.num_layers](x, e0)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)

    @staticmethod
    def convert_from_original(original_state_dict, num_layers):
        """Convert original WanModel state_dict to WanModelInterleaved format.

        Mapping (for layer i):
          Original                              → Interleaved
          blocks.i.norm1.*                      → units.{i if i==0 else i}.attn_half.norm1.*
          blocks.i.self_attn.*                  → units.{i if i==0 else i}.attn_half.self_attn.*
          blocks.i.norm3.*                      → units.{i if i==0 else i}.attn_half.norm3.*
          blocks.i.cross_attn.*                 → units.{i if i==0 else i}.attn_half.cross_attn.*
          blocks.i.norm2.*                      → units.{i+1 if i<N-1 else N}.ffn_half.norm2.*
          blocks.i.ffn.*                        → units.{i+1 if i<N-1 else N}.ffn_half.ffn.*
          blocks.i.modulation (1,6,dim)         → split into attn_half.modulation (1,3,dim)
                                                  and ffn_half.modulation (1,3,dim)

        Unit layout:
          units.0 = WanFirstUnit:     layer0.attn_half
          units.i = WanStaggeredUnit: layer[i-1].ffn_half + layer[i].attn_half   (i=1..N-1)
          units.N = WanLastUnit:      layer[N-1].ffn_half
        """
        new_sd = {}
        N = num_layers

        attn_keys = ['norm1.', 'self_attn.', 'norm3.', 'cross_attn.']
        ffn_keys = ['norm2.', 'ffn.']

        for key, value in original_state_dict.items():
            if not key.startswith('blocks.'):
                new_sd[key] = value
                continue

            parts = key.split('.', 2)
            layer_idx = int(parts[1])
            rest = parts[2]

            if rest == 'modulation':
                # shape [1, 6, dim] → split into [1, 3, dim] each
                attn_mod = value[:, :3, :]
                ffn_mod = value[:, 3:, :]

                # attn_half modulation → unit for this layer's attn
                if layer_idx == 0:
                    attn_unit_idx = 0
                    new_sd[f'units.{attn_unit_idx}.attn_half.modulation'] = attn_mod
                else:
                    attn_unit_idx = layer_idx
                    new_sd[f'units.{attn_unit_idx}.attn_half.modulation'] = attn_mod

                # ffn_half modulation → unit for this layer's ffn
                if layer_idx < N - 1:
                    ffn_unit_idx = layer_idx + 1
                    new_sd[f'units.{ffn_unit_idx}.ffn_half.modulation'] = ffn_mod
                else:
                    ffn_unit_idx = N
                    new_sd[f'units.{ffn_unit_idx}.ffn_half.modulation'] = ffn_mod
                continue

            is_attn = any(rest.startswith(k) for k in attn_keys)
            is_ffn = any(rest.startswith(k) for k in ffn_keys)

            if is_attn:
                if layer_idx == 0:
                    new_sd[f'units.0.attn_half.{rest}'] = value
                else:
                    new_sd[f'units.{layer_idx}.attn_half.{rest}'] = value
            elif is_ffn:
                if layer_idx < N - 1:
                    new_sd[f'units.{layer_idx + 1}.ffn_half.{rest}'] = value
                else:
                    new_sd[f'units.{N}.ffn_half.{rest}'] = value

        return new_sd
