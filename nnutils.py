import torch

# A file containing common utility functions for neural network modeling in PyTorch.

#Why? Because I was getting tired of copy-pasting the same code over and over again.

## Utility functions included in this file:
# 1. RMSNorm
# 2. GeGLU
# 3. ReluSquared
# 4. MLP
# 5. CausalConv1d
# 6. ModernizedLSTM

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

class MLP(torch.nn.Module):
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
    
class ModernizedLSTM(torch.nn.Module):
    def __init__(self, indims):
        super().__init__()
        self.indims = indims
        self.lstm = torch.nn.LSTM(indims, indims)
        self.norm = RMSNorm(indims)
        self.projection = torch.nn.Linear(indims, indims)
        self.r_projection = torch.nn.Linear(indims, indims, bias=False)
        self.q_projection = torch.nn.Linear(indims, indims, bias=False)
    def forward(self, x, hidden=None):
        res = x
        x = self.norm(x)
        q = self.q_projection(x)
        r = torch.sigmoid(self.r_projection(x))

        x, hidden = self.lstm(q, hidden)
        x = x * r

        x = self.projection(x)

        return x, hidden