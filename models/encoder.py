"""
This file defines the encoder architecture
"""
import torch
import torch.nn as nn
from models.DRSformer import subnet, OverlapPatchEmbed
import pdb


class ResBlock(nn.Module):
    def __init__(self, C_in, C_out):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_in, kernel_size=3, stride=1, padding=1)
        self.se1 = SEBlock(C_in, reduction=16)
        self.conv2 = nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1)
        self.se2 = SEBlock(C_out, 16)
        self.relu = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        residual = x
        out = self.se1(self.conv1(x))
        out = self.relu(out)
        out = self.se2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, C_in, reduction):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [b, c, 1, 1]
            nn.Conv2d(C_in, C_in // reduction, 1, 1, 0, bias=False),
            nn.ReLU(),
            nn.Conv2d(C_in // reduction, C_in, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.se(x)
        return att * x


class RainAtoms(nn.Module):
    """
    obtain rain chromatic, degradation, location atoms
    """
    def __init__(self, dim, base_dim):
        super(RainAtoms, self).__init__()
        self.chromatic_enc = nn.Sequential(
            nn.Conv2d(dim, base_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, False),
            ResBlock(base_dim, base_dim)
        )
        self.degra_enc = nn.Sequential(
            nn.Conv2d(dim, base_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, False),
            ResBlock(base_dim, base_dim)
        )
        self.spatial_pool = nn.AvgPool2d(kernel_size=8, stride=8)  # pool for contrastive learning
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # pool for atom representation learning

    def forward(self, x):
        _, _, h, w = x.shape
        chromatic_feat = self.chromatic_enc(x)
        degra_feat = self.degra_enc(x)
        pool_chromatic = self.avg_pool(chromatic_feat)  # vector
        pool_detail = self.spatial_pool(degra_feat) # 2D
        pool_degra = self.avg_pool(degra_feat) # vector

        z_chromatic = nn.functional.normalize(self.avg_pool(chromatic_feat), dim=1)
        z_degra = nn.functional.normalize(self.avg_pool(degra_feat), dim=1)
        return pool_chromatic, pool_detail, pool_degra, z_chromatic, z_degra

class Encoder(nn.Module):
    def __init__(self, dim, base_dim, size=128):
        super(Encoder, self).__init__()
        self.conv = OverlapPatchEmbed(in_c=3, embed_dim=dim, bias=True)
        self.MoEs = subnet(dim=dim)  # mixture of experts
        self.Atom = RainAtoms(dim, base_dim)
        self.size = size
    
    def forward(self, x):
        _, _, h, w = x.size()
        x = self.conv(x)
        feat_atom = self.MoEs(x)
        # feat_atom_inter = nn.functional.interpolate(feat_atom, (self.size, self.size), mode='bicubic')
        pool_chromatic, pool_detail, pool_degra, z_chromatic, z_degra = self.Atom.forward(feat_atom)
        out_dict = {
            "location": feat_atom,
            "chromatic": (z_chromatic, pool_chromatic),
            "degradation": (z_degra, pool_degra),
            "detail": (z_degra, pool_detail),
        }
        return out_dict
    
    def get_loc(self, x):
        x = self.conv(x)
        loc = self.MoEs(x)
        return loc

class LocalKnowledgeFusion(nn.Module):
    def __init__(self, dim):
        super(LocalKnowledgeFusion, self).__init__()
        self.location_map = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, False),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.channel_map = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.LeakyReLU(0.1, False),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, loc_feat, channel_feat):
        """
        loc_feat from encoder: provide location information
        channel_feat from model: provide channel information
        """
        loc_map = self.location_map(loc_feat)
        loc_map = loc_map.mean(dim=1, keepdims=True) # [B, 1, H, W]
        loc_enhance_feat = torch.sigmoid(loc_map * channel_feat) * loc_feat # [B, C, H, W]
        # process channel
        channel_map = self.avg_pool(channel_feat) # [B, C, 1, 1]
        channel_map = self.channel_map(channel_map)
        channel_enhance_feat = torch.sigmoid(channel_map * loc_feat) * channel_feat
        return loc_enhance_feat + channel_enhance_feat