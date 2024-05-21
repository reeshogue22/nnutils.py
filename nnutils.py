import torch

# A file containing common utility functions for neural network modeling in PyTorch.

#Why? Because I was getting tired of copy-pasting the same code over and over again.

## Utility functions included in this file:
# 1. RMSNorm
# 2. GeGLU
# 3. ReluSquared
# 4. GeGLUMLP
# 5. CausalConv1d
# 6. GatedAttention

class RMSNorm(torch.nn.Module):
    def __init__(self, indims, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.scale = torch.nn.Parameter(torch.ones(indims))
    def forward(self, x):
        return x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon) * self.scale)
    
class GeGLU(torch.nn.Module):
    def __init__(self, indims, hiddims):
        super().__init__()
        self.proj = torch.nn.Linear(indims, hiddims * 2, bias=False)
    def forward(self, x):
        x1, x2 = self.proj(x).chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)    

class ReluSquared(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)*torch.relu(x)

class GeGLUMLP(torch.nn.Module):
    def __init__(self, indims, hiddims, outdims):
        super().__init__()
        self.geglu = GeGLU(indims, hiddims)
        self.out = torch.nn.Linear(hiddims, outdims, bias=False)
    def forward(self, x):
        x = self.geglu(x)
        x = self.out(x)
        return x
    
class CausalConv1d(torch.nn.Module):
    def __init__(self, indims, hiddims, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv1d(indims, hiddims, kernel_size)
    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.conv.kernel_size[0]-1, 0))
        x = self.conv(x)
        return x
    
class GatedAttention(torch.nn.Module):
    def __init__(self, indims, nheads, causal=False):
        super().__init__()
        self.qkvr = torch.nn.Linear(indims, indims*4, bias=False)
        self.proj = torch.nn.Linear(indims, indims, bias=False)

        self.causal = causal
        self.nheads = nheads
    def forward(self, x):
        assert x.dim() == 3
        q, k, v, r = self.qkvr(x).chunk(4, dim=-1)
        q = q.view(*q.size()[:-1], self.nheads, q.size(-1)//self.nheads).transpose(1, 2) # (batch, nheads, seq, head)
        k = k.view(*k.size()[:-1], self.nheads, k.size(-1)//self.nheads).transpose(1, 2) # (batch, nheads, seq, head)
        v = v.view(*v.size()[:-1], self.nheads, v.size(-1)//self.nheads).transpose(1, 2) # (batch, nheads, seq, head)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).view(*attn.size()[:-2], attn.size(-2)*attn.size(-1))
        out = attn * torch.sigmoid(r)
        out = self.proj(out)
        return out
