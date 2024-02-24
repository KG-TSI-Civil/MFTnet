"""
@Reference: https://github.com/raoyongming/GFNet/blob/master/gfnet.py; https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB023657
"""

import torch
import torch.nn as nn
import torch.fft


class PatchEmbed(nn.Module):
    def __init__(self, seq_len,  projection_dim):
        super(PatchEmbed, self).__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, projection_dim))
        nn.init.trunc_normal_(self.positional_embedding, std=.02)

    def forward(self, x):
        embedded = x +  self.positional_embedding  
        return embedded


class GlobalFilter(nn.Module): # Substitution of standard attention layer
    def __init__(self,sequence, dim): # hidden_dim must be (input dimension/2 +1)
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(sequence, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x): #  B * sequencelength * channel
        x = torch.fft.rfft2(x, dim=(-2,-1), norm='ortho') # B * sequencelength * (channel/2 +1)
        weight = torch.view_as_complex(self.complex_weight) 
        x = x * weight  # B * sequencelength * (channel/2 +1)
        x = torch.fft.irfft2(x, dim=(-2, -1), norm='ortho') # B * sequencelength * channel
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
      

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNet(nn.Module):  # Fourier Transformer
    def __init__(self, sequence, dim, depth, mlp_dim, dropout = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, GlobalFilter(sequence,int(dim/2 +1))),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MFTnet(nn.Module):
    def __init__(self,seq_len=24,conv_dim=[3,32,64,128],projection_dim=128,transformer_layers=4,ff_size=128*2,drop_rate=0.1,num_predictions=4,atten_type='FFT'):
        """ 
        Args:
            seq_len (int): the input sequence length of transformer
            conv_dim (list): dimension of each convolution layer
            projection_dim (int): the input dimension of transformer, the same as conv_dim[-1]
            ff_size (int): middle dimension of mlp in transformer layer 
            num_predictions (int): Number of source parameters to be predicted. 4 for magnitude, epicentral distance, p travel time, depth
            atten_type (str): 'Attention' for using the standard transformer encoder, 'FFT' for using the Fourier Transformer  

        Returns:
            list of tensor, shape of each tensor is: [batch_size, 1]
        """
        super(MFTnet, self).__init__()

        # Convolution for size reduction:    
        conv_layers = [
            [nn.Conv1d(conv_dim[0],conv_dim[1],3,2,padding=1),nn.Dropout(drop_rate),nn.MaxPool1d(4,4)],  
            [nn.Conv1d(conv_dim[1],conv_dim[2],3,2,padding=1),nn.Dropout(drop_rate),nn.MaxPool1d(2,2)],
            [nn.Conv1d(conv_dim[2],conv_dim[3],3,2,padding=2),nn.Dropout(drop_rate),nn.MaxPool1d(2,2)]
        ]
        conv_modules = []
        for conv_layer in conv_layers:
            conv_modules.extend(conv_layer)
        self.ConvBlocks = nn.Sequential(*conv_modules)        


        # Learnable patch embedding:
        self.patch_embed = PatchEmbed(seq_len=seq_len,
                                      projection_dim=projection_dim)


        # Transformer layers for feature extraction:
        if atten_type =="Attention":
            encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim,
                                                       nhead=4,
                                                       dim_feedforward=ff_size,
                                                       dropout=drop_rate,
                                                       layer_norm_eps=1e-6,
                                                       batch_first=True,
                                                       norm_first=True)

            self.Transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        else:
            self.Transformer= FNet(seq_len,projection_dim,transformer_layers,ff_size, 0.1)


        self.post_norm = nn.LayerNorm(projection_dim, eps=1e-6)


        # Decoders:
        conv_layers = []
        mlp_layers = []
        for _ in range(num_predictions):
            conv_layer = nn.Sequential(
                nn.Conv1d(projection_dim, 32, 1, 1),
                nn.BatchNorm1d(32),
                nn.GELU()
            )
            conv_layers.append(conv_layer)

            mlp_layer = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(768, 1)
            )
            mlp_layers.append(mlp_layer)

        self.post_convs = nn.ModuleList(conv_layers)
        self.mlps = nn.ModuleList(mlp_layers)


    def forward(self, x):
        outputs=[]
        x = self.ConvBlocks(x) # shape: b,32,375 -> b,64,47 -> b,128,24
        x = x.permute(0,2,1) # b,24,128
        x = self.patch_embed(x)  # b,24,128
        x = self.Transformer(x)   # b,24,128
        x = self.post_norm(x)   # b,24,128
        x = x.permute(0,2,1) # b,128,24
        for conv, mlp in zip(self.post_convs, self.mlps):
            x_conv = conv(x)
            x_mlp = mlp(x_conv.flatten(1))
            outputs.append(x_mlp)
        return outputs
    

if __name__ == "__main__":
    projection_dim = 128
    transformer_layers = 4
    ff_size = projection_dim * 2
    drop_rate=0.1
    seq_len=24
    conv_dim = [3,32,64,128]
    num_prediction = 4

    model=MFTnet(seq_len=seq_len,conv_dim=conv_dim,projection_dim=projection_dim,transformer_layers=transformer_layers, ff_size=ff_size,drop_rate=drop_rate,num_predictions=num_prediction,atten_type='Attention')

    x = torch.rand(2,3,3000)
    y = model(x)
    print(y)

