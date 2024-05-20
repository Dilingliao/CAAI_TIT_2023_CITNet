from email.policy import strict
from multiprocessing import pool
from re import S
from telnetlib import SE
from tokenize import group
from xmlrpc.client import INVALID_ENCODING_CHAR
import PIL
import time
from cv2 import norm
from importlib_metadata import re
from numpy import identity, reshape
from sklearn.model_selection import GroupShuffleSplit
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import _adaptive_avg_pool2d, conv2d, nn, strided
import torch.nn.init as init
import math
import numpy as np
import matplotlib.pyplot as plt

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, dim)
        self.GELU = GELU()
        self.Dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.Linear1(x)
        x = self.GELU(x)
        x = self.Dropout(x)
        x = self.Linear2(x)
        x = self.Dropout(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv2d(65,65,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(65)
        self.GELU = GELU()


    def forward(self, x, mask=None):       # x=(64,5,64)
        b, n, _, h = *x.shape, self.heads
        # print('x3',x1.shape)
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        # k = k + x2
        # q,k,v  (64, 8, 65, 8)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # print(dots.shape)
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out1 = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax


        # print(out1.shape)
        out = rearrange(v, 'b h n d -> b n h d')
        out = self.conv(out)
        out = self.bn(out)
        # out = self.GELU(out)
        # out = self.GELU(out)
        out = rearrange(out, 'b n h d -> b h n d')
        # print(out.shape)
        out = out + out1
        # out = rearrange(out, 'b n h d -> b h n d')


        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.attention = Attention(dim, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP_Block(dim, mlp_dim, dropout=dropout)

    def forward(self, x1, x2, mask=None):
        # for attention, mlp in self.layers:
        identity = x1         # (64,65,64)
        # print(identity.shape)
        x1 = self.norm(x1)
        x1 = self.attention(x1, mask=mask)  # go to attention   [64, 65, 64]
        # print(x1.shape)
        x1 = x1 + identity
        x22 = self.norm(x1)
        x22 = self.mlp(x22)  # go to MLP_Block
        x = x22 + x1
        return x, x1



BATCH_SIZE_TRAIN = 32
NUM_CLASS = 16

def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=5, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # x1 = visualize_feature_map(avg_pool)
                # print('1',avg_pool.shape)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        # Gauss modulation
        mean = torch.mean(channel_att_sum).detach()
        std = torch.std(channel_att_sum).detach()
        scale = GaussProjection(channel_att_sum, mean, std).unsqueeze(-1).unsqueeze(-1).expand_as(x)

        x = x * scale + x

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)




class CITNet(nn.Module):
    def __init__(self, xta, xte, xall, xy, num_classes=16, in_channels=1, band=30, num_tokens=30, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1,reduction_ratio=10, pool_types=['avg']):
        super(CITNet, self).__init__()

        self.name = 'CITNet'

        self.L = num_tokens
        self.cT = dim
        # define
        # self.Conv3d_attn = nn.Conv3d(in_channels, out_channels=12, kernel_size=(30, 1, 1))
        # self.Conv2d_attn = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=13)
        self.ChannelGate = ChannelGate(64, 5, pool_types)

        self.conv3d_features1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(7, 7, 7), padding=(3,3,3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv3d_features2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=64, kernel_size=(1, 1, 30), padding=(0,0,0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=64*1, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization
        
        self.token_wA = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, self.L, band),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)

        self.token_wA1 = nn.Parameter(torch.empty(xta % BATCH_SIZE_TRAIN, self.L, band),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA1)
        
        self.token_wA2 = nn.Parameter(torch.empty(xte % BATCH_SIZE_TRAIN, self.L, band),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA2)

        self.token_wA3 = nn.Parameter(torch.empty(xall % BATCH_SIZE_TRAIN, self.L, band),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA3)

        self.token_wV = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, band, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)
       
        self.token_wV1 = nn.Parameter(torch.empty(xta % BATCH_SIZE_TRAIN, band, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV1)
        
        self.token_wV2 = nn.Parameter(torch.empty(xte % BATCH_SIZE_TRAIN, band, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV2)

        self.token_wV3 = nn.Parameter(torch.empty(xall % BATCH_SIZE_TRAIN, band, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV3)

        self.pos_embedding = nn.Parameter(torch.empty(1, (dim + 1), dim))
        # self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, 1))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
  

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(64, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.patch_to_embedding = nn.Linear(xy, dim)

        self.pooling2d = nn.AdaptiveAvgPool2d(1)
        self.Conv = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.GELU = GELU()
        self.sigmoid = nn.Sigmoid() 
        self.Linear = nn.Linear(64, 64)
 
    def forward(self, x, mask=None):   

        x = self.conv3d_features1(x)
        x = self.conv3d_features2(x)
        # print(x.shape)
        x = rearrange(x, 'b c h w y -> b c y h w')
        # identity = x
        x = rearrange(x, 'b c h w y -> b (c h) w y')   
        identity = x
        x = self.ChannelGate(x)
        x = self.conv2d_features(x)   

        x = rearrange(x, 'b c h w -> b c (h w)')     

        x = self.patch_to_embedding(x)     #[b,n,dim]
        b, n, _ = x.shape        # 

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) 
        x = torch.cat((cls_tokens, x), dim = 1)

        x += self.pos_embedding[:, :(n + 1)] 
        x = self.dropout(x)    

        x, x2 = self.transformer(x, identity, mask)
        x = self.to_cls_token(x[:, 0])     

        x = self.nn1(x)


        return x


if __name__ == '__main__':
    model = CITNet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 11, 11)   
    y = model(input)
    print(y.size())