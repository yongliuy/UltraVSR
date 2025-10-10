import torch
import torch.nn as nn
from einops import rearrange
import math
import copy

def batchlist2fulltensor(x_batch_list):
    x = []
    for i in range(len(x_batch_list)):
        x.append(x_batch_list[i])
    x = torch.cat(x, dim=0)

    return x

def fulltensor2batchlist(x, inference_num_frames_per_batch):
    t_, c_, h_, w_ = x.shape
    bs_inference = math.ceil(t_/(inference_num_frames_per_batch))
    x_batch_list = []
    for i in range(bs_inference):
        x_batch_list.append(x[i*inference_num_frames_per_batch:min((i+1)*inference_num_frames_per_batch, t_), :, :, :])

    return x_batch_list

class RTSConvBlock(nn.Module):
    def __init__(self, nhidden=128, bias=False, LoRA_ratio=10):
        super().__init__()
        nhidden_ = nhidden//(LoRA_ratio*3)*3 ## 3 = 1+2*order

        self.conv1 = nn.Conv2d(nhidden, nhidden_, 1, 1, 0, bias=bias)

        self.conv_blocks = nn.ModuleList([])
        self.conv_blocks.append(nn.Conv2d(nhidden_, nhidden_, 3, 1, 1, bias=True))
        self.conv_blocks.append(nn.SiLU())

        self.conv3 = nn.Conv2d(nhidden_, nhidden, 1, 1, 0, bias=bias)
        nn.init.zeros_(self.conv3.weight)
        
    def shift(self, x, reverse=False):
        ## x: T N C H W
        assert x.shape[2] % 3 == 0, "The channel of input tensor should be divided by 3."
        fold = x.shape[2] // 3
        x_shifted = torch.zeros_like(x)
        if not reverse:
            # +1
            x_shifted[1:, :, :fold, :, :] = x[:-1, :, :fold, :, :]
            ## -1
            x_shifted[:-1, :, -1*fold:, :, :] = x[1:, :, -1*fold:, :, :]
        else:
            # +1
            x_shifted[:-1, :, :fold, :, :] = x[1:, :, :fold, :, :]
            ## -1
            x_shifted[1:, :, -1*fold:, :, :] = x[:-1, :, -1*fold:, :, :]     

        return x_shifted.contiguous()

    def RTS_shift(self, x, bs, reverse=None):
        x = rearrange(x, "(N T) C H W -> T N C H W", N=bs)
        x = self.shift(x, reverse=reverse)
        x = rearrange(x, "T N C H W -> (N T) C H W")
        return x

    def forward(self, x, bs, inference_num_frames_per_batch):
        ## x: (N T) C H W
        if inference_num_frames_per_batch is None:
            x_pre = x
            x = self.conv1(x)
            x = self.RTS_shift(x, bs)
            for block in self.conv_blocks:
                x = block(x)
            # x = self.RTS_shift(x, bs, reverse=True)
            x = self.conv3(x)
            x = x + x_pre
        else:
            x_pre = copy.deepcopy(x)
            bs_inference = len(x)
            for i in range(bs_inference):
                x_ = self.conv1(x[i])
                x[i] = x_
            x = batchlist2fulltensor(x)
            x = self.RTS_shift(x, bs)
            x = fulltensor2batchlist(x, inference_num_frames_per_batch)
            for i in range(bs_inference):
                x_ = x[i]
                for block in self.conv_blocks:
                    x_ = block(x_)
                x_ = self.conv3(x_)
                x[i] = x_ + x_pre[i]

        return x

class RTSAttenBlock(nn.Module):
    def __init__(self, dim, num_heads=8, LoRA_ratio=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        dim_atten = dim//(LoRA_ratio*num_heads*3)*(num_heads*3)

        self.qkv = nn.Conv2d(dim, dim_atten, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim_atten, dim_atten, kernel_size=3, stride=1, padding=1, groups=dim_atten, bias=True)
        self.project_out = nn.Conv2d(dim_atten//3, dim, kernel_size=1, bias=bias)
        nn.init.zeros_(self.project_out.weight)
    
    def shift(self, x, strade):
        ## x: T N head c len
        # x_shifted = x.clone()
        x_shifted = torch.zeros_like(x)
        if strade == 1:
            x_shifted[1:, :, :, :, :] = x[:-1, :, :, :, :]
        elif strade == -1:
            x_shifted[:-1, :, :, :, :] = x[1:, :, :, :, :]

        return x_shifted.contiguous()
    

    def forward(self, x, bs, inference_num_frames_per_batch):
        ## x: (N T) C H W
        ####
        if inference_num_frames_per_batch is None:
            x_pre = x
            b,c,h,w = x.shape
            qkv = self.qkv_dwconv(self.qkv(x))
        else:
            x_pre = copy.deepcopy(x)
            bs_inference = len(x)
            b,c,h,w = x[0].shape
            for i in range(bs_inference):
                x_ = self.qkv_dwconv(self.qkv(x[i]))
                x[i] = x_
            qkv = batchlist2fulltensor(x)

        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        k = rearrange(k, "(N T) head c len -> T N head c len", N=bs).contiguous()
        v = rearrange(v, "(N T) head c len -> T N head c len", N=bs).contiguous()
        
        out1 = self.attention(q, k, v, strade=1, bs=bs)
        out2 = self.attention(q, k, v, strade=0, bs=bs)
        out3 = self.attention(q, k, v, strade=-1, bs=bs)
        del q, k, v

        out = out1 + out2 + out3
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        del out1, out2, out3

        if inference_num_frames_per_batch is None:
            out = self.project_out(out)
            out = out + x_pre
        else:
            out = fulltensor2batchlist(out, inference_num_frames_per_batch)
            for i in range(bs_inference):
                out_ = self.project_out(out[i])
                out[i] = out_ + x_pre[i]
        return out

    def attention(self, q, k, v, strade, bs):
        k = self.shift(k, strade=strade)
        v = self.shift(v, strade=strade)
        k = rearrange(k, "T (N L) head c len -> (N T L) head c len", N=bs)
        v = rearrange(v, "T (N L) head c len -> (N T L) head c len", N=bs)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out

class RTSModule(nn.Module):
    def __init__(
        self,
        nhidden=128,
    ):
        super().__init__()
        self.RTSConv = RTSConvBlock(nhidden)
        self.RTSAtten = RTSAttenBlock(nhidden)

    def forward(self, x, bs=1, inference_num_frames_per_batch=None, device=None):
        if isinstance(x, list) and device is not None:
            ori_device = x[0].device
            for i in range(len(x)):
                x[i] = x[i].to(device)

        x = self.RTSConv(x, bs, inference_num_frames_per_batch)
        x = self.RTSAtten(x, bs, inference_num_frames_per_batch)

        if isinstance(x, list) and device is not None:
            for i in range(len(x)):
                x[i] = x[i].to(ori_device)

        return x
