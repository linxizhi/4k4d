import torch.nn as nn
import torch
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, D=8, W=64, input_ch=32, output_ch=4, skips=[4],activation=nn.ReLU,out_activation=None):
        """ 
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch

        self.skips = skips
        linears=[]
        linears.append(nn.Linear(input_ch, W))
        for i in range(D-1):
            input_channe,out_channel=W,W
            linears.append(nn.Linear(input_channe,out_channel))
        self.linears=nn.ModuleList(linears)
        self.output_linear=nn.Linear(W,output_ch)
        self.activation=activation()
        if out_activation is not None:
            self.output_activation=out_activation()
        else:
            self.output_activation=None


    def forward(self,features):
        h = features
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = self.activation(h)
            if i in self.skips:
                h = torch.cat([features, h], -1)
        outputs = self.output_linear(h)
        if self.output_activation is not None:
            outputs=self.output_activation(outputs)
        return outputs  
    
class SH_MLP(MLP):
    def __init__(self, D=8, W=64, input_ch=32, degree=3, skips=[4], activation=nn.Softplus, out_activation=None):
        output_ch = (degree + 1) ** 2
        output_dim=output_ch*3
        self.output_ch=output_ch
        super().__init__(D, W, input_ch, output_dim, skips, activation, out_activation)
    def forward(self,features):
        out= super().forward(features)
        out=out.reshape(-1,3,self.output_ch)
        return out