import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tools.smart_exchange import img_se

import ipdb


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *kargs,**kwargs):
        out,head_score=self.fn(x, *kargs,**kwargs)
        return out+ x,head_score

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *kargs,**kwargs):
        out,head_score = self.fn(self.norm(x), *kargs,**kwargs)
        return out,head_score

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x),self.hidden_dim

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qkv_bias = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.head_score = torch.ones(heads)
#        self.attn_score = nn.Parameter(torch.rand(4,heads,197,197))

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        #self.token_pruning = token_pruning()


    def forward(self, x, mask = False):#这里加参数，atten_score
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        #mask_value = -torch.finfo(dots.dtype).max

        # if mask is not None: #不用的mask
        #     mask = F.pad(mask.flatten(1), (1, 0), value = True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
        #     dots.masked_fill_(~mask, mask_value)
        #     del mask

        attn = dots.softmax(dim=-1)
        #TODO: 将每一层的Attention得分append进一个数组，然后传出去
        #token_attn_score = torch.sum(one_head_score,dim=1)#(4,1,1,197)
        #token spatten operation:
        #attn_score = attn_score.cuda() + attn
        #one_head_score=torch.sum(attn_score, dim=-2)#(4,12,1,197)
        #token_attn_score = torch.sum(one_head_score,dim=1)#(4,1,1,197)
        #token_attn_score_sort,_ = token_attn_score.sort(1,descending = True)
        #threshold = torch.sum(token_attn_score_sort[:,150])/4#稀疏度设置：25%
        #token_mask = torch.gt(token_attn_score,threshold)
        # spar = torch.sum(token_mask,dim=-1)
        # spar = torch.sum(spar)
        # print(spar/(4*197))
        #token_mask = token_mask.reshape(4,197,1).expand(4,197,768)   

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        head_score = torch.sum(torch.abs(out),dim=(2,3)) 
        head_score = torch.mean(head_score,dim=0)
        #self.head_score = head_score.softmax(-1)
        self.head_score=torch.nn.functional.normalize(head_score,p=1,dim=0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        #out = out*token_mask
        out =  self.to_out(out)
        return out,self.head_score

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qkv_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qkv_bias=qkv_bias))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, mask = False):
        head_scores=[]
        #attn_score = torch.zeros((4,12,197,197))
        for attn, ff in self.layers:
            x,head_score = attn(x, mask = mask)
            x,_ = ff(x)
            head_scores.append(head_score)
        return x,head_scores

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., qkv_bias=False):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # nn.Linear(patch_dim, dim),
            # use conv2d to fit weight file
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qkv_bias)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = False):
        # img[b, c, img_h, img_h] > patches[b, p_h*p_w, dim]
        x = self.to_patch_embedding(img)
        x = x.flatten(2).transpose(1,2)
        # ipdb.set_trace()
        b, n, _ = x.shape

        # cls_token[1, p_n*p_n*c] > cls_tokens[b, p_n*p_n*c]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # add(concat) cls_token to patch_embedding
        x = torch.cat((cls_tokens, x), dim=1)
        # add pos_embedding
        x += self.pos_embedding[:, :(n + 1)]
        # drop out
        x = self.dropout(x)

        # main structure of transformer
        if mask:
            None
#            x,_= img_se(x)    #input SE

        x ,head_scores= self.transformer(x, mask)

        # use cls_token to get classification message
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x),head_scores
