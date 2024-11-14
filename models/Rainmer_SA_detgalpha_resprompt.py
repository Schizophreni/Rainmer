# DRSformer with knowledge atoms:  All atoms are required
# detail (3D), degradation (1D), localtion (2D), and illumination (1D) atoms
# knowledge injection occurs at the decoder stages (attention MCA part) 
# serial attention edition, sparse attention first, prompt attention second
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from functools import reduce
from models.modules import MoCo_RainAtom as MoCo
from fractions import Fraction
import time
import pdb
import torchvision
import math
from models.encoder import Encoder, LocalKnowledgeFusion


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
    
    # @torch.compile()
    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

class PromptAttention(nn.Module):
    # prompt attention
    def __init__(self, dim, base_dim):
        super(PromptAttention, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.trans = nn.Sequential(
            nn.Conv2d(base_dim, dim, 1, 1, 0),
            nn.SiLU(),
        )
    
    def forward(self, x, prompt):
        gate = self.classifier(x)  # [b, 1, h, w]
        prompt_val = self.trans(gate * prompt) # [b, 1, h, w] * [b, d, 1, 1] -> [b, d, h, w]
        return prompt_val

##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, sparse_rates=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.sparse_rates = sparse_rates
        if self.sparse_rates:
            self.attns = torch.nn.Parameter(torch.ones(len(self.sparse_rates))*0.2, requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # sparse attention mechanism
        if self.sparse_rates:
            outs = []
            for rate_idx, rate in enumerate(self.sparse_rates):
                rate = Fraction(rate)
                mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                index = torch.topk(attn, k=int(C * rate), dim=-1, largest=True)[1]
                mask.scatter_(-1, index, 1.)
                attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
                attn = attn.softmax(dim=-1)
                outs.append((attn @ v) * self.attns[rate_idx])
            out = reduce(torch.add, outs)
        else:
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

##  cross attention Sparse Attention (TKSA)
# @torch.compile()
class CrossAttention(nn.Module):
    def __init__(self, dim, base_dim, num_heads, bias, sparse_rates=None):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.base_dim = base_dim

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.sparse_rates = sparse_rates
        if self.sparse_rates:
            self.attns = torch.nn.Parameter(torch.ones(len(self.sparse_rates))*0.2, requires_grad=True)
        self.chr_promptAttn = PromptAttention(dim, base_dim)
        self.detg_promptAttn = PromptAttention(dim, base_dim)
        self.detg_z = torch.nn.Parameter(torch.ones(1, base_dim, 1, 1, dtype=torch.float32), requires_grad=True)

    # @torch.compile()
    def forward(self, x, global_knowledge):
        b, c, h, w = x.shape 
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, H, C, C] * [B, H, 1, 1]
        # sparse attention mechanism
        if self.sparse_rates:
            outs = []
            for rate_idx, rate in enumerate(self.sparse_rates):
                rate = Fraction(rate)
                mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                index = torch.topk(attn, k=int(C * rate), dim=-1, largest=True)[1]
                mask.scatter_(-1, index, 1.)
                attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
                attn = attn.softmax(dim=-1)
                outs.append((attn @ v) * self.attns[rate_idx])  # [B, H, C, L] * [B, H, 1, 1]
            out = reduce(torch.add, outs)
        else:
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out) 
        #################################
        # prompt attention (spatial attention)
        #################################
        if global_knowledge is not None:
            if len(global_knowledge) == 1:
                prompt_out = self.chr_promptAttn(out, global_knowledge[0])
            else:
                prompt_chr = self.chr_promptAttn(out, global_knowledge[0])
                detg_alpha = F.normalize(self.detg_z) * global_knowledge[1].detach()
                detg_alpha = torch.sum(detg_alpha, dim=1, keepdims=True) # [b, 1, 1, 1]
                prompt_detg = self.detg_promptAttn(out, global_knowledge[1])
                prompt_out = (1-detg_alpha) * prompt_chr + detg_alpha * prompt_detg
            return out + prompt_out
        else:
            return out

##  Sparse Transformer Block (STB)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, sparse_rates):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, sparse_rates)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##  Cross attention Sparse Transformer Block (STB)
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, base_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, sparse_rates,
                 prompt_attention=False):
        super(CrossTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, sparse_rates)
        self.chr_promptAttn = PromptAttention(dim, base_dim)
        self.detg_promptAttn = PromptAttention(dim, base_dim)
        self.detg_z = torch.nn.Parameter(torch.ones(1, base_dim, 1, 1, dtype=torch.float32), requires_grad=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x, global_knowledge = x
        x = x + self.attn(self.norm1(x))
        residual = x
        # prompt attn
        if global_knowledge is not None:
            if len(global_knowledge) == 1:
                prompt_out = self.chr_promptAttn(x, global_knowledge[0])
            else:
                prompt_chr = self.chr_promptAttn(x, global_knowledge[0])
                detg_alpha = F.normalize(self.detg_z) * global_knowledge[1].detach()
                detg_alpha = torch.sum(detg_alpha, dim=1, keepdims=True) # [b, 1, 1, 1]
                prompt_detg = self.detg_promptAttn(x, global_knowledge[1])
                prompt_out = (1-detg_alpha) * prompt_chr + detg_alpha * prompt_detg
            x = residual + prompt_out
        x = x + self.ffn(self.norm2(x))
        return (x, global_knowledge)

class OperationLayer(nn.Module):
    def __init__(self, C, stride):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(nn.Conv2d(C * len(Operations), C, 1, padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1, 0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x) * w.view([-1, 1, 1, 1]).contiguous())
        return self._out(torch.cat(states[:], dim=1))


class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()
        self.preprocess = ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(C, stride)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0


class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Conv2d(channel, self.output * 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(self.output * 2, self.k * self.num_ops, 1, 1, 0))

    def forward(self, x):
        y = self.avg_pool(x)
        # y = y.view(x.size(0), -1).contiguous()
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops).contiguous()
        return y


Operations = [
    'sep_conv_1x1',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7',
    'avg_pool_3x3'
]

OPS = {
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'sep_conv_1x1': lambda C, stride, affine: SepConv(C, C, 1, stride, 0, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_conv_7x7': lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
}


class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


## Mixture of Experts Feature Compensator (MEFC)
# @torch.compile()
class subnet(nn.Module):
    def __init__(self, dim, layer_num=1, steps=4):
        super(subnet, self).__init__()

        self._C = dim
        self.num_ops = len(Operations)
        self._layer_num = layer_num
        self._steps = steps

        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(steps, self._C)
            self.layers += [layer]
    
    # @torch.compile()
    def forward(self, x):

        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(x)
                weights = F.softmax(weights, dim=-1)
            else:
                x = layer(x, weights)

        return x


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))  # channel: n_feat // 2 * 4 = 2*n_feat

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))  # channel: n_feat * 2 // 4 = n_feat // 2

    def forward(self, x):
        return self.body(x)


class Rainmer(nn.Module):
    def __init__(self, opt):
        super(DRSformer, self).__init__()
        inp_channels, out_channels, dim = opt.inp_channels, opt.out_channels, opt.dim
        base_dim, num_blocks, heads = opt.base_dim, opt.num_blocks, opt.heads
        ffn_expansion_factor, bias, LayerNorm_type = opt.ffn_expansion_factor, opt.bias, opt.LayerNorm_type
        sparse_rates = opt.sparse_rates
        proj_types = opt.knowledge_atoms
        print(proj_types)
        self.base_dim = base_dim
        self.norm_scale = math.sqrt(base_dim)
        # define encoder
        self.opt = opt
        self.disentangle = MoCo(encoder_q=Encoder(dim, base_dim), 
                              encoder_k=Encoder(dim, base_dim),
                              temperature=1.0,
                              m=0.999)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level0 = subnet(dim)

        self.cross_fuse = LocalKnowledgeFusion(dim)
        
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, sparse_rates=sparse_rates) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, sparse_rates=sparse_rates) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, sparse_rates=sparse_rates) for i in range(num_blocks[2])])
        
        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, sparse_rates=sparse_rates) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            CrossTransformerBlock(dim=int(dim * 2 ** 2), base_dim=base_dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, sparse_rates=sparse_rates) for i in range(num_blocks[2])])
        
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            CrossTransformerBlock(dim=int(dim * 2 ** 1), base_dim=base_dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, sparse_rates=sparse_rates) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            CrossTransformerBlock(dim=int(dim * 2 ** 1), base_dim=base_dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, sparse_rates=sparse_rates) for i in range(num_blocks[0])])

        self.refinement = subnet(dim=int(dim * 2 ** 1))  ## We do not use MEFC for training Rain200L and SPA-Data

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        cnt = 0
        for param in self.parameters():
            if param.requires_grad:
                cnt += param.numel()
        print("total parameters: ", cnt)

    def forward(self, inp_img, im_q, im_k_dict, im_negs_dict, adapt=False):
        """
        im_k_dict: ["illu", "detail", "degradation"]
        one x_k, n_neg x_neg 
        """
        knowledge_dict = dict()
        # distangle atomic knowledge
        if adapt:
            out_dict, logits = self.disentangle.forward(latent_feat_q=im_q,
                                                    latent_k_dict=im_k_dict,
                                                    latent_negs_dict=im_negs_dict)
            # location_feat = out_dict["location"]
            global_knowledge = []
            if "chromatic" in self.opt.knowledge_atoms:
                global_knowledge.append(out_dict["chromatic"])
            if "degradation" in self.opt.knowledge_atoms:
                global_knowledge.append(out_dict["degradation"])
            if len(global_knowledge) == 0:
                global_knowledge = None
        else:
            global_knowledge = None
            logits = None

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level0 = self.encoder_level0(inp_enc_level1)
        # obtain location feat
        location_feat = self.disentangle.encoder_q.get_loc(inp_img) 
        inp_enc_level0 = self.cross_fuse.forward(loc_feat=location_feat, channel_feat=inp_enc_level0)

        out_enc_level1 = self.encoder_level1(inp_enc_level0)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3((inp_dec_level3, global_knowledge))

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2((inp_dec_level2, global_knowledge))

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, _ = self.decoder_level1((inp_dec_level1, global_knowledge))

        out_dec_level1 = self.refinement(out_dec_level1)  ## We do not use MEFC for training Rain200L and SPA-Data

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, logits
    
    def encode(self, inp_img, im_q, im_k_dict, im_negs_dict):
        """
        im_k_dict: ["chromatic", "degradation"]
        one x_k, n_neg x_neg 
        """
        # accelerate encode process by wraping x, x_k and x_negs together
        _, logits = self.disentangle.forward(latent_feat_q=im_q, latent_k_dict=im_k_dict,
                                           latent_negs_dict=im_negs_dict)
        return logits

    @torch.no_grad()
    def inference(self, inp_img, im_q):
        knowledge_dict = dict()
        # distangle atomic knowledge
        self.eval()
        out_dict = self.disentangle.encoder_q.forward(im_q)

        global_knowledge = []
        if "chromatic" in self.opt.knowledge_atoms:
            global_knowledge.append(out_dict["chromatic"][0])
        if "degradation" in self.opt.knowledge_atoms:
            global_knowledge.append(out_dict["degradation"][0])
        if len(global_knowledge) == 0:
            global_knowledge = None
        
        b = inp_img.shape[0]

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level0 = self.encoder_level0(inp_enc_level1) 
        location_feat = self.disentangle.encoder_q.get_loc(inp_img)
        
        inp_enc_level0 = self.cross_fuse.forward(loc_feat=location_feat, channel_feat=inp_enc_level0)

        out_enc_level1 = self.encoder_level1(inp_enc_level0)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3((inp_dec_level3, global_knowledge))

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2((inp_dec_level2, global_knowledge))

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, _ = self.decoder_level1((inp_dec_level1, global_knowledge))

        out_dec_level1 = self.refinement(out_dec_level1)  ## We do not use MEFC for training Rain200L and SPA-Data

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from utils.parse_config import parse
    opt = parse()
    input = torch.rand(1, 3, 96, 96)
    im_k_dict = {"chromatic": torch.rand(1, 3, 96, 96),
                 "degradation": torch.rand(1, 3, 96, 96),
                 "detail": torch.rand(1, 3, 96, 96)}
    im_neg_dict = {"chromatic": torch.rand(1, 2, 3, 96, 96),
                   "degradation": torch.rand(1, 2, 3, 96, 96),
                   "detail": torch.rand(1, 2, 3, 96, 96)}
    model = DRSformer(opt.model)
    with torch.no_grad():
        output = model.forward(inp_img=input, im_k_dict=im_k_dict, im_negs_dict=im_neg_dict)
        print(output[0].shape)
        print("logits: ", output[-1]["degradation"])