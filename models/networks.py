import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfm.model.dfm import DeepFactorizationMachineModel

##############################
# DeepFactorizationMachine
##############################
DeepFM = DeepFactorizationMachineModel

##############################
# base attention
##############################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialAttn(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(SpatialAttn, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(in_planes, out_planes, kernel_size, stride=1, padding=0, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttn(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttn, self).__init__()
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
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


##############################
# ATT-CNN module
##############################

class AttentionNetwork(nn.Module):
    def __init__(self):
        super(AttentionNetwork, self).__init__()
        kernel_size = (1,32)
        self.ms1 = SpatialAttn(2, 1, kernel_size)
        kernel_size = (1, 1)
        self.ms2 = SpatialAttn(2, 1, kernel_size)
        self.mc = ChannelAttn(gate_channels=32*3)
        # l is the length of the time window # 3month
        # w is the dimension of the row vector in the input matrix. 32
        lw1_kernel_size = (3, 32)
        lw2_kernel_size = (5, 32)
        lw3_kernel_size = (10, 32)
        
        self.convlw1 = BasicConv(1, 32, lw1_kernel_size, stride=1, padding=(1,0), relu=False,  bn=True)
        self.convlw2 = BasicConv(1, 32, lw2_kernel_size, stride=1, padding=(2,0), relu=False,  bn=True)
        self.convlw3 = BasicConv(1, 32, lw3_kernel_size, stride=1, padding=(4,0), relu=False,  bn=True)
        # self.convlw = BasicConv(1, 32, lw_kernel_size, stride=1, padding=0, relu=False,  bn=True)
        w1_kernel_size = (5, 1)
        self.convw1 = BasicConv(96, 32, w1_kernel_size, stride=1, padding=(2,0), relu=False,  bn=True)

    def forward(self, x):
        x1 = self.ms1(x) # x1.size == x.size as (1, 1, 8760, 32)
        p1 = self.convlw1(x1) # (1, 32, 8760, 1)
        p2 = self.convlw2(x1) # (1, 32, 8760, 1)
        p3 = self.convlw3(x1) # (1, 32, 8760, 1)
        p3 = torch.cat((p3, torch.Tensor(1, 32, 1, 1)), dim=2)
        p = torch.cat( (p1, p2, p3), dim=1) # (1, 96, 8760, 1)
        
        p = self.mc(p) # p = (1, 96, 8760, 1)
        p = self.ms2(p) # p = (1, 96, 8760, 1)
        output = self.convw1(p) #  ([1, 32, 8760, 1])
        return output

class FACNN(nn.Module):
    def __init__(self):
        field_dims, embed_dim, mlp_dims, dropout = 1, 2, 3, 4
        self.fa = DeepFM(field_dims, embed_dim, mlp_dims, dropout)
        self.att_cnn = AttentionNetwork()
        #全結合層
        self.dence = nn.Sequential(
            nn.Linear('わからない', 64),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        x1 = self.fa(x)
        x2 = self.att_cnn(x)
        # x1 + x2 結合
        # full connected layer 
        # output 









