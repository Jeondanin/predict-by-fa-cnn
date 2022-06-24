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
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(torch.Tensor.float(x),1).unsqueeze(1)), dim=1 )

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
    def __init__(self, gate_channels, reduction_ratio=1, pool_types=['avg', 'max']):
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
        self.mc = ChannelAttn(gate_channels=2)
        # l is the length of the time window # 2month
        # w is the dimension of the row vector in the input matrix. 32
        lw1_kernel_size = (3, 32)
        lw2_kernel_size = (5, 32)
        lw3_kernel_size = (10, 32)
        
        self.convlw1 = BasicConv(1, 1, lw1_kernel_size, stride=1, padding=(1,0), relu=False,  bn=True)
        self.convlw2 = BasicConv(1, 1, lw2_kernel_size, stride=1, padding=(2,0), relu=False,  bn=True)
        # self.convlw3 = BasicConv(1, 1, lw3_kernel_size, stride=1, padding=(4,0), relu=False,  bn=True)
        # self.convlw = BasicConv(1, 32, lw_kernel_size, stride=1, padding=0, relu=False,  bn=True)
        w1_kernel_size = (3, 1)
        self.convw1 = BasicConv(1*2, 1, w1_kernel_size, stride=1, padding=(1,0), relu=False,  bn=True)

    def forward(self, x):
        x1 = self.ms1(x) # x1.size == x.size as (batch, 1, dayrange, 32)
        p1 = self.convlw1(x1) # (batch, 32, 1, range)
        p2 = self.convlw2(x1) # (1, 32, 8760, 1)
        # p3 = self.convlw3(x1) # (1, 32, 8760, 1)
        # p3 = torch.cat((p3, torch.Tensor(1, 32, 1, 1)), dim=2)
        p = torch.cat( (p1, p2), dim=1) # (1, 96, 8760, 1)
        
        p = self.mc(p) # p = ([10, 2, 1, 1])
        p = self.ms2(p) # p =([10, 2, 1, 1])
        output = self.convw1(p) # ([10, 32, 1, 1])
        return output

class FACNN(nn.Module):
    def __init__(self):
        """
        field_dims : the number of index and time
            [[t1, t2, ,tm], [i1_1, i1_2, i1_n]]
            [[t1, t2, ,tm], [i2_1, i2_2, i2_n]]
            [[t1, t2, ,tm], [i3_1, i3_2, i3_n]]
            [[t1, t2, ,tm], [i4_1, i4_2, i4_n]]
            [[t1, t2, ,tm], [i5_1, i5_2, i5_n]]
            eaach index; may be 33
        embed_dim : embed次元（適当に）
        """
        super().__init__()
        field_dims, embed_dim, mlp_dims, dropout = tuple(32+i for i in range(32)) ,16, [32, 16, 8], 0.2
        self.fa = DeepFM(field_dims, embed_dim, mlp_dims, dropout)
        self.att_cnn = AttentionNetwork()
        #全結合層
        self.dence = nn.Sequential(
            nn.Linear(2, 64),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )
    
    def forward(self, x):
        
        # x1 = [self.fa(x[i]) for i in range(len(x))]

        # example
        #######input x shape: (2048, 34) - (batch?, field_dims)
        #######field_dims: 34
        #######num_factors: 10
        #######mlp_dims: [30, 20, 10]
        ####### self.embedding = nn.Embedding(num_inputs, num_factors) - (3816, 10)
        #######output x shape: (2048, 1) - (batch, 1)
        
        # my case
        
        #######field_dims: 32
        #######num_factors: 16
        
        #######output x shape: (1, 1)

        # faのinput shape: (1, 32)
        x1 = [self.fa(i) for i in x]
        x2 = self.att_cnn(x.unsqueeze(1).unsqueeze(1)) # [batch, 32, 1, 1]
        c_x2 = x2.squeeze(2).squeeze(2) # [batch, 32] [10 32]
        # x1 + x2 結合
        stack_x1 = torch.stack(x1)  # ([10, 1])
        
        zx = torch.cat((stack_x1, c_x2), dim=1) # (10,2)
        # full connected layer 
        output = self.dence(zx)  # 
        return output









