"""
This file defines basic loss functions required
basic loss:
contrastive loss: degradation, illumination, and detail
location prediction loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


# MoCo with cosine similarity
class MoCo_cos(nn.Module):
    """
    Build MoCo model with: query encoder, key encoder (momentum update)
    """
    def __init__(self, encoder_q, encoder_k, m=0.999, temperature=1.0) -> None:
        super(MoCo_cos, self).__init__()
        self.m = m
        self.temperature = temperature
        self.encoder_q, self.encoder_k = encoder_q, encoder_k
        # copy params
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    def forward(self, latent_feat_q, latent_k_dict, latent_negs_dict):
        """
        latent_feat_q: query latent features: [b, D, h, w]
        latent_feat_neg: neg latent features: [b, n_neg, D, h, w]
        """
        atom_types = list(latent_k_dict.keys())
        latent_feat_ks, latent_feat_negs = [], []
        n_atom = len(atom_types)
        atom = None
        logits_dict = dict()
        # process latent_q
        out_dict_q = self.encoder_q(latent_feat_q)
        b, dim, h, w = latent_feat_q.shape

        # process latent_k_dict and latent_negs_dict simultaneously
        for atom in atom_types:
            latent_feat_ks.append(latent_k_dict[atom])
            latent_feat_negs.append(latent_negs_dict[atom])
        latent_feat_ks = torch.cat(latent_feat_ks, dim=0)  # [n_atom * b, c, fh, fw]
        n_neg = latent_negs_dict[atom].shape[1]
        latent_feat_negs = torch.cat(latent_feat_negs, dim=0).reshape(n_atom*b*n_neg, dim, h, w).contiguous() # [n_atom * b, n_neg, c, fh, fw]
        latent_feat_kns = torch.cat([latent_feat_ks, latent_feat_negs], dim=0)

        # parse f_k and f_negs
        # update encoder_k
        self._momentum_update_key_encoder()
        # encode key and neg images
        out_dict_kn = self.encoder_k(latent_feat_kns)
        # parse each atomic knowledge
        for idx, atom in enumerate(atom_types):
            z_q, proj_q = out_dict_q[atom]
            _, emb_dim, fh, fw = proj_q.shape  # obtain feature projected size of feature
            proj_ks, proj_negs = out_dict_kn[atom][1][:b*n_atom], out_dict_kn[atom][1][b*n_atom:].view(b * n_atom, n_neg, emb_dim, fh, fw).contiguous()
            proj_k = proj_ks[idx*b : (idx+1)*b]  # obtain that atom key features, shape: [b, c, h, w]
            proj_neg = proj_negs[idx*b : (idx+1)*b]  # obtain that atom neg features, shape [b, n_neg, c, h, w]
            proj_q = F.normalize(proj_q, dim=1)
            proj_k = F.normalize(proj_k, dim=1)  # normalize
            proj_neg = F.normalize(proj_neg, dim=2)  # normalize
            proj_q = proj_q.view(b, emb_dim * fh * fw).contiguous()
            proj_k = proj_k.view(b, emb_dim * fh * fw).contiguous()
            proj_neg = proj_neg.view(b, n_neg, emb_dim * fh * fw).contiguous()
            # pixel-wise contrastive logits
            l_pos = (proj_q * proj_k).sum(dim=-1, keepdims=True) / (fh * fw)
            l_neg = (proj_q.unsqueeze(1) * proj_neg).sum(dim=-1) / (fh * fw)
            logits = torch.cat([l_pos, l_neg], dim=-1) / self.temperature
            logits_dict[atom] = logits  # calculate logits for that atom
            out_dict_q[atom] = z_q
            # print(logits)
            # print(atom, z_q.shape)
            # pdb.set_trace()
        return out_dict_q, logits_dict
    
# MoCo with cosine similarity
class MoCo_RainAtom(nn.Module):
    """
    Build MoCo model with: query encoder, key encoder (momentum update)
    """
    def __init__(self, encoder_q, encoder_k, m=0.999, temperature=1.0) -> None:
        super(MoCo_RainAtom, self).__init__()
        self.m = m
        self.temperature = temperature
        self.encoder_q, self.encoder_k = encoder_q, encoder_k
        # copy params
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    def forward(self, latent_feat_q, latent_k_dict, latent_negs_dict):
        """
        keys: detail, chromatic, degradation
        latent_feat_q: query latent features: [b, D, h, w]
        latent_feat_neg: neg latent features: [b, n_neg, D, h, w]
        """
        atom_types = list(latent_k_dict.keys())
        latent_feat_ks, latent_feat_negs = [], []
        n_atom = len(atom_types)
        atom = None
        logits_dict = dict()
        # process latent_q
        out_dict_q = self.encoder_q(latent_feat_q)
        b, dim, h, w = latent_feat_q.shape

        # process latent_k_dict and latent_negs_dict simultaneously
        for atom in atom_types:
            latent_feat_ks.append(latent_k_dict[atom])
            latent_feat_negs.append(latent_negs_dict[atom])
        latent_feat_ks = torch.cat(latent_feat_ks, dim=0)  # [n_atom * b, c, fh, fw]
        n_neg = latent_negs_dict[atom].shape[1]
        latent_feat_negs = torch.cat(latent_feat_negs, dim=0).reshape(n_atom*b*n_neg, dim, h, w).contiguous() # [n_atom * b, n_neg, c, fh, fw]
        latent_feat_kns = torch.cat([latent_feat_ks, latent_feat_negs], dim=0)

        # parse f_k and f_negs
        # update encoder_k
        self._momentum_update_key_encoder()
        # encode key and neg images
        out_dict_kn = self.encoder_k(latent_feat_kns)
        # parse each atomic knowledge
        for idx, atom in enumerate(atom_types):
            z_q, proj_q = out_dict_q[atom]
            _, emb_dim, fh, fw = proj_q.shape  # obtain feature projected size of feature
            proj_ks, proj_negs = out_dict_kn[atom][1][:b*n_atom], out_dict_kn[atom][1][b*n_atom:].reshape(b * n_atom, n_neg, emb_dim, fh, fw).contiguous()
            proj_k = proj_ks[idx*b : (idx+1)*b]  # obtain that atom key features, shape: [b, c, h, w]
            proj_neg = proj_negs[idx*b : (idx+1)*b]  # obtain that atom neg features, shape [b, n_neg, c, h, w]
            proj_q = F.normalize(proj_q, dim=1)
            proj_k = F.normalize(proj_k, dim=1)  # normalize
            proj_neg = F.normalize(proj_neg, dim=2)  # normalize
            proj_q = proj_q.reshape(b, emb_dim * fh * fw).contiguous()
            proj_k = proj_k.reshape(b, emb_dim * fh * fw).contiguous()
            proj_neg = proj_neg.reshape(b, n_neg, emb_dim * fh * fw).contiguous()
            # pixel-wise contrastive logits
            l_pos = (proj_q * proj_k).sum(dim=-1, keepdims=True) / (fh * fw)
            l_neg = (proj_q.unsqueeze(1) * proj_neg).sum(dim=-1) / (fh * fw)
            logits = torch.cat([l_pos, l_neg], dim=-1) / self.temperature
            logits_dict[atom] = logits  # calculate logits for that atom
            out_dict_q[atom] = z_q
            # print(logits)
            # print(atom, z_q.shape)
            # pdb.set_trace()
        return out_dict_q, logits_dict