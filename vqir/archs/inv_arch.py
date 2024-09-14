import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from archs.layers import *

class ICRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ICRM, self).__init__()
        self.in_nc = in_channels         #change from 3 to 1
        self.out_nc = out_channels
        self.operations = nn.ModuleList()

        ###########################################
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)

        # 256->128->64->32->16
        for i in range(4):
            b = Conv1x1(self.in_nc, self.in_nc//2)
            self.operations.append(b)
            self.in_nc //= 2
            b = InvertibleConv1x1(self.in_nc)
            self.operations.append(b)
            b = AttModule(self.in_nc, self.in_nc//2)
            self.operations.append(b)
            ###########################################
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)

        # 16->8->3
        self.inpfeat_extraction = nn.ModuleList()

        b = Conv1x1(self.in_nc, self.in_nc//2)
        self.inpfeat_extraction.append(b)
        self.in_nc //=2
        b = InvertibleConv1x1(self.in_nc)
        self.inpfeat_extraction.append(b)
        b = AttModule(self.in_nc, self.in_nc//2)
        self.inpfeat_extraction.append(b)

        b = Conv1x1(self.in_nc, self.out_nc)
        self.inpfeat_extraction.append(b)


    def forward(self, x, rev=False):
        if not rev:
            for op in self.operations:
                x = op.forward(x, False)
            for op in self.inpfeat_extraction:
                x = op.forward(x, False)
        else:
            for op in reversed(self.inpfeat_extraction):
                x = op.forward(x, True)
            for op in reversed(self.operations):
                x = op.forward(x, True)
        return x
    
class ICRM_32(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ICRM_32, self).__init__()
        self.in_nc = in_channels         #change from 3 to 1
        self.out_nc = out_channels
        self.operations = nn.ModuleList()

        ###########################################
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)

        # 256->128->64->32->16
        for i in range(4):
            b = Conv1x1(self.in_nc, self.in_nc//2)
            self.operations.append(b)
            self.in_nc //= 2
            b = InvertibleConv1x1(self.in_nc)
            self.operations.append(b)
            b = AttModule(self.in_nc, self.in_nc)
            self.operations.append(b)
            ###########################################
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)

        b = HaarDownsampling(self.in_nc)
        self.operations.append(b)
        self.in_nc *= 4
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        ###########################################
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
        self.operations.append(b)

        # 64->32->16
        for i in range(2):
            b = Conv1x1(self.in_nc, self.in_nc//2)
            self.operations.append(b)
            self.in_nc //= 2
            b = InvertibleConv1x1(self.in_nc)
            self.operations.append(b)
            b = AttModule(self.in_nc, self.in_nc)
            self.operations.append(b)
            ###########################################
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)
            b = CouplingLayer(self.in_nc // 2, self.in_nc // 2, 3)
            self.operations.append(b)

        # 16->8->3
        self.inpfeat_extraction = nn.ModuleList()

        b = Conv1x1(self.in_nc, self.in_nc//2)
        self.inpfeat_extraction.append(b)
        self.in_nc //=2
        b = InvertibleConv1x1(self.in_nc)
        self.inpfeat_extraction.append(b)
        b = AttModule(self.in_nc, self.in_nc)
        self.inpfeat_extraction.append(b)

        b = Conv1x1(self.in_nc, self.out_nc)
        self.inpfeat_extraction.append(b)


    def forward(self, x, rev=False):
        if not rev:
            for op in self.operations:
                x = op.forward(x, False)
            for op in self.inpfeat_extraction:
                x = op.forward(x, False)
        else:
            for op in reversed(self.inpfeat_extraction):
                x = op.forward(x, True)
            for op in reversed(self.operations):
                x = op.forward(x, True)
        return x
    
class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac
    
class AttModule(nn.Module):
    def __init__(self, N, M):
        super(AttModule, self).__init__()
        self.forw_att = AttentionBlock(N, M)
        self.back_att = AttentionBlock(N, M)
        
    def forward(self, x, rev = True):
        if not rev:
            return self.forw_att(x)
        else:
            return self.back_att(x)

class CouplingLayer(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp
        # print(split_len1,split_len2)

        self.G1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(x2)) * 2 - 1) )) + self.H2(x2)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(y1)) * 2 - 1) )) + self.H1(y1)
        else:
            y2 = (x2 - self.H1(x1)).div(torch.exp( self.clamp * (torch.sigmoid(self.G1(x1)) * 2 - 1) ))
            y1 = (x1 - self.H2(y2)).div(torch.exp( self.clamp * (torch.sigmoid(self.G2(y2)) * 2 - 1) ))
        return torch.cat((y1, y2), 1)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        initialize_weights(self.conv3, 0)
        
    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        return conv3

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, reverse=False):
        weight = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            #print(z.size())
            return z
        else:
            z = F.conv2d(input, weight)
            return z
        
class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,1)
        self.conv2 = nn.Conv2d(out_channels,in_channels,1)

    def forward(self, input, reverse=False):
        if not reverse:
            z = self.conv1(input)
            #print(z.size())
            return z
        else:
            z = self.conv2(input)
            return z

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
    x = torch.randn(4,256,32,32)
    model = ICRM(in_channels=256,out_channels=3)
    y = model(x, False)
    z = model(y, True)
    print(y.shape,z.shape)