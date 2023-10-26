
"""
@CreatedDate:   2023/06
@Author: Katherine_Cao(https://github.com/Katherine-Cao/HSI_SNN)
@used: Li-SNN(
"""
import math

import torch
import torch.nn as nn
import numpy as np
class MixConv2d(nn.Module):
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """
        :params c1: 
        :params c2: 
        :params k: 
        :params s: 
        :params equal_ch: 
        """
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

   
    def forward(self, x):
        
        return self.bn(torch.cat([m(x) for m in self.m], 1))

class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
     
        temp = torch.abs(1 - torch.abs(torch.arcsin(input))) < 0.7
        return grad_input * temp.float()

def channel_shuffle(x, groups: int):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels 
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    
    x = x.view(batch_size, -1, height, width)
    return x

class TGRS(nn.Module):
    def __init__(self, num_steps, leak_mem, img_size, num_cls, input_dim):
        super(TGRS, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem


        bias_flag = False

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)

       
        self.conv1_1 = MixConv2d(64, 64, k=(1, 3), s=1, equal_ch=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.conv1_3 = MixConv2d(64, 64, k=(1, 3), s=1, equal_ch=True)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.pool1 = nn.AvgPool2d(kernel_size=2)

     
        self.conv3_1 = MixConv2d(128, 128, k=(1, 3), s=1, equal_ch=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=bias_flag)
        self.conv3_3 = MixConv2d(128, 128, k=(3, 5), s=1, equal_ch=True)
        self.conv3_4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=bias_flag)

      
        self.conv5_1 = MixConv2d(128, 128, k=(3, 5), s=1, equal_ch=True)
        self.conv5_2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=bias_flag)
        self.conv5_3 = MixConv2d(128, 128, k=(3, 5), s=1, equal_ch=True)
        self.conv5_4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=bias_flag)

        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, self.num_cls, bias=False) 

        self.conv_list = [self.conv1, self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4, self.conv2, self.conv3_1, self.conv3_2,
                          self.conv3_3, self.conv3_4, self.conv4, self.conv5_1, self.conv5_2,
                          self.conv5_3, self.conv5_4, self.conv6]
   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)


    def forward(self, input):
        batch_size = input.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv1_1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv1_2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv1_3 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv1_4 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv3_1 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3_2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3_3 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3_4 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv5_1 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv5_2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv5_3 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv5_4 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_fc1 = torch.zeros(batch_size, self.num_cls).cuda()

        mem_conv_list = [mem_conv1, mem_conv1_1, mem_conv1_2, mem_conv1_3, mem_conv1_4, mem_conv2, mem_conv3_1,
                         mem_conv3_2, mem_conv3_3, mem_conv3_4, mem_conv4, mem_conv5_1, mem_conv5_2,
                         mem_conv5_3, mem_conv5_4, mem_conv6]

        static_input1 = self.conv1(input)
        def forward(self, index, input_value):
            mem_conv_list[index] = self.leak_mem * mem_conv_list[index] + (1 - self.leak_mem) * self.conv_list[index](input_value) 
            mem_thr = mem_conv_list[index] - 1
            out = self.spike_fn(mem_thr)     
            rst = torch.zeros_like(mem_conv_list[index]).cuda()
            rst[mem_thr > 0] = 1
            mem_conv_list[index] = mem_conv_list[index] - rst
            out_prev = out.clone()
            return out_prev

        for t in range(self.num_steps):
            mem_conv_list[0] = self.leak_mem * mem_conv_list[0] + (1 - self.leak_mem) * static_input1  # 总分支
            mem_thr = mem_conv_list[0] - self.conv_list[0].threshold
            out = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[0]).cuda()
            rst[mem_thr > 0] = self.conv_list[0].threshold
            mem_conv_list[0] = mem_conv_list[0] - rst
            out1 = out.clone()

         
            value2 = forward(self, 1, out1)
            value3 = forward(self, 2, value2)
            value4 = forward(self, 3, out1)
          
            value5 = forward(self, 4, value4)

        
            shortcut1 = value3 + value5 + out1
            value6 = forward(self, 5, shortcut1)
            pool1 = self.pool1(value6)

            value7 = forward(self, 6, pool1)
          
            value8 = forward(self, 7, value7)
            value9 = forward(self, 8, pool1)
          
            value10 = forward(self, 9, value9)

         
            shortcut2 = pool1 + value8 + value10
            value11 = forward(self, 10, shortcut2)  

            value12 = forward(self, 11, value11)
           
            value13 = forward(self, 12, value12)
            value14 = forward(self, 13, value11)
         
            value15 = forward(self, 14, value14)

         
            shortcut3 = value13 + value15 + shortcut2
            value16 = forward(self, 15, shortcut3)
            pool2 = self.pool2(value16)

            out_prev = pool2.reshape(batch_size, -1)
            mem_fc1 = mem_fc1 + self.fc1(out_prev)
        out_voltage = mem_fc1 / self.num_steps

        return out_voltage
