"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['Scattering']

import warnings
import torch
from utils import cdgmm, Modulus, Periodize, Fft
from filters_bank import filters_bank
from torch.nn import ReflectionPad2d as pad_function

from skimage import io
import numpy as np
import cv2
import os

class Scattering_Gabor9(object):
    """Scattering module.
       Gabor变换完整版，加上第0层低通。
    """
    #M = 32, N = 32, J = 2, pre_pad=False
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering_Gabor9, self).__init__()
        print('Scattering_Gabor9'.center(80, '='))
        self.M, self.N, self.J = M, N, J   
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit) #模数
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)#每个方向的填充长度=4

        # Create the filters
        #返回16个小波函数以及2个尺度函数（J=2，L=8）
        filters = filters_bank(self.M_padded, self.N_padded, J) #(40,40,2)

        #16个小波函数
        self.Psi = filters['psi']
        #2个尺度函数
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J  #40
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J  #40

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)#fill_(0)就是new一个真正全零的tensor。
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module(input)#每个方向的填充长度=4
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)#这里面的2应该就是从实数转为复数的意思
            output.narrow(4, 0, 1).copy_(out_.unsqueeze(4))#复制给第4个维度从0到1索引
        return output

    #最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要
    def _unpad(self, in_):        		
        return in_[..., 4:-4, 4:-4]

    #input = (16,3,32,32)
    def forward(self, input):
        #NCHW
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft #fft包装器
        periodize = self.periodize #划分时期
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      9,
                      32,
                      32)               #new的这个是一个全趋于零的tensor，知识type和device都与inputs保持一致。
        
        #torch.Size([16, 3, 40, 40, 2])
        U_r = pad(input)#copies and view the real to complex
        
        #进行傅里叶变换
        #torch.Size([16, 3, 40, 40, 2])
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
        
        # First low pass filter（）
        #torch.Size([16, 3, 10, 10, 2])
        U_1_c = cdgmm(U_0_c, phi[0], jit=self.jit)                                                            #这里加低通

        #torch.Size([16, 3, 10, 10])
        U_J_r = fft(U_1_c, 'C2R')#傅里叶反变换,将输入的最后一维去掉，即去掉2.

        S[..., n, :, :].copy_(unpad(U_J_r))

        n = n + 1#n=1,第2个channel
        
        #psi[n1]一共有16个，每一个都是一个字典，属性为j和theta,0三个，其中0的值就是滤波器（张量）
        #结束后n=81（(j-1)+...+1=64）
        for n1 in range(len(psi)):#len(psi)=16
            j1 = psi[n1]['j']
            #U_0_c:torch.Size([16, 3, 40, 40, 2]);psi[n1][0]:(40*40*2)
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)#进行小波变换后：torch.Size([16, 3, 40, 40, 2])
            if(j1 > 0):
                break
				#torch.Size([16, 3, 20, 20, 2]),尺寸除以2
                U_1_c = periodize(U_1_c, k=2 ** j1)                                               #试试不进行FT反变换

            U_J_r = fft(U_1_c, 'C2R')#torch.Size([16, 3, 10, 10])
            S[..., n, :, :].copy_(unpad(U_J_r))#unpad(U_J_r):torch.Size([16, 3, 8, 8])
            n = n + 1

        if (n != 9):
            raise (RuntimeError('this is a debug'))            
        

        return S#输出：torch.Size([16, 3, 8, 32, 32])

    ##该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, input):
        return self.forward(input)


class Scattering_Gabor2_9(object):
    """Scattering module.
    因为Gabor2_9(不包括第0层)的resnet结果最起码训练到一半的时候结果真好，所以保留一下。
    这个类的代码也可以用来被剪切波的滤波器替换，因为剪切波的滤波器中就直接包含低通结果，所以不用包含第0层。
    just_shearlet
    """
    #M = 32, N = 32, J = 2, pre_pad=False
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering_Gabor2_9, self).__init__()
        print('Scattering_Gabor2_9'.center(80, '='))
        self.M, self.N, self.J = M, N, J   
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit) #模数
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)#每个方向的填充长度=4

        # Create the filters
        #返回16个小波函数以及2个尺度函数（J=2，L=8）
        filters = filters_bank(self.M_padded, self.N_padded, J) #(40,40,2)

        #16个小波函数
        self.Psi = filters['psi']
        #2个尺度函数
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J  #40
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J  #40

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)#fill_(0)就是new一个真正全零的tensor。
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module(input)#每个方向的填充长度=4
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)#这里面的2应该就是从实数转为复数的意思
            output.narrow(4, 0, 1).copy_(out_.unsqueeze(4))#复制给第4个维度从0到1索引
        return output

    #最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要
    def _unpad(self, in_):        		
        return in_[..., 4:-4, 4:-4]

    #input = (16,3,32,32)
    def forward(self, input):
        #NCHW
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft #fft包装器
        periodize = self.periodize #划分时期
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      9,
                      32,
                      32)               #new的这个是一个全趋于零的tensor，知识type和device都与inputs保持一致。
        
        #torch.Size([16, 3, 40, 40, 2])
        U_r = pad(input)#copies and view the real to complex
        
        #进行傅里叶变换
        #torch.Size([16, 3, 40, 40, 2])
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
        
        #psi[n1]一共有16个，每一个都是一个字典，属性为j和theta,0三个，其中0的值就是滤波器（张量）
        #结束后n=81（(j-1)+...+1=64）
        for n1 in range(len(psi)):#len(psi)=16
            j1 = psi[n1]['j']
            #U_0_c:torch.Size([16, 3, 40, 40, 2]);psi[n1][0]:(40*40*2)
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)#进行小波变换后：torch.Size([16, 3, 40, 40, 2])
            if(j1 > 0):
                break
				#torch.Size([16, 3, 20, 20, 2]),尺寸除以2
                U_1_c = periodize(U_1_c, k=2 ** j1)
            # fft(U_1_c, 'C2C', inverse=True, inplace=True)#傅里叶反变换                                                    #试试不进行FT反变换

            U_J_r = fft(U_1_c, 'C2R')#torch.Size([16, 3, 10, 10])
            S[..., n, :, :].copy_(unpad(U_J_r))#unpad(U_J_r):torch.Size([16, 3, 8, 8])
            n = n + 1

        if (n != 9):
            raise (RuntimeError('this is a debug'))            
        

        return S#输出：torch.Size([16, 3, 8, 32, 32])

    ##该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, input):
        return self.forward(input)
        

class Scattering_shearlet_py(object):
    """Scattering module.
    这个和上面说的剪切波不一样在于这里的滤波器是python代码，而上面说的是matlab代码产生的滤波器。
    
    """
    #M = 32, N = 32, J = 2, pre_pad=False
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering_shearlet_py, self).__init__()
        print('Scattering_shearlet_py'.center(80, '='))
        self.M, self.N, self.J = M, N, J   
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit) #模数
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)#每个方向的填充长度=4

        # Create the filters
        #返回16个小波函数以及2个尺度函数（J=2，L=8）
        filters = filters_bank(self.M_padded, self.N_padded, J) #(40,40,2)

        #16个小波函数
        self.Psi = filters['psi']
        #2个尺度函数
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J  #40
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J  #40

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)#fill_(0)就是new一个真正全零的tensor。
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module(input)#每个方向的填充长度=4
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)#这里面的2应该就是从实数转为复数的意思
            output.narrow(4, 0, 1).copy_(out_.unsqueeze(4))#复制给第4个维度从0到1索引
        return output

    #最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要
    def _unpad(self, in_):        		
        return in_[..., 4:-4, 4:-4]

    #input = (16,3,32,32)
    def forward(self, input):
        #NCHW
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft #fft包装器
        periodize = self.periodize #划分时期
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      13,
                      32,
                      32)               #new的这个是一个全趋于零的tensor，知识type和device都与inputs保持一致。
        
        #torch.Size([16, 3, 40, 40, 2])
        U_r = pad(input)#copies and view the real to complex
        
        #进行傅里叶变换
        #torch.Size([16, 3, 40, 40, 2])
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
        
        #psi[n1]一共有16个，每一个都是一个字典，属性为j和theta,0三个，其中0的值就是滤波器（张量）
        #结束后n=81（(j-1)+...+1=64）
        for n1 in range(len(psi)):#len(psi)=16
            j1 = psi[n1]['j']
            #U_0_c:torch.Size([16, 3, 40, 40, 2]);psi[n1][0]:(40*40*2)
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)#进行小波变换后：torch.Size([16, 3, 40, 40, 2])
            if(j1 > 0):
                break
				#torch.Size([16, 3, 20, 20, 2]),尺寸除以2
                U_1_c = periodize(U_1_c, k=2 ** j1)
            # fft(U_1_c, 'C2C', inverse=True, inplace=True)#傅里叶反变换                                                    #试试不进行FT反变换

            U_J_r = fft(U_1_c, 'C2R')#torch.Size([16, 3, 10, 10])
            S[..., n, :, :].copy_(unpad(U_J_r))#unpad(U_J_r):torch.Size([16, 3, 8, 8])
            n = n + 1

        if (n != 13):
            raise (RuntimeError('this is a debug'))            
        

        return S#输出：torch.Size([16, 3, 8, 32, 32])

    ##该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, input):
        return self.forward(input)



class Scattering_no_downsample(object):
    """Scattering module.

    Runs scattering on an input image in NCHW（N代表数量， C代表channel，H代表高度，W代表宽度） format

    Input args:
        M, N: input image size
        J: number of layers
        pre_pad: if set to True, module expect pre-padded images
        jit: compile kernels on the fly for speed（快速编译核）
    """
    #M = 32, N = 32, J = 2, pre_pad=False
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering_no_downsample, self).__init__()
        print('Scattering_no_downsample'.center(80, '='))
        self.M, self.N, self.J = M, N, J   
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit) #模数
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)#每个方向的填充长度=4

        # Create the filters
        #返回16个小波函数以及2个尺度函数（J=2，L=8）
        filters = filters_bank(self.M_padded, self.N_padded, J) #(40,40,2)

        #16个小波函数
        self.Psi = filters['psi']
        #2个尺度函数
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J  #40
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J  #40

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)#fill_(0)就是new一个真正全零的tensor。
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module(input)#每个方向的填充长度=4
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)#这里面的2应该就是从实数转为复数的意思
            output.narrow(4, 0, 1).copy_(out_.unsqueeze(4))#复制给第4个维度从0到1索引
        return output

    #最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要
    def _unpad(self, in_):        		
        return in_[..., 4:-4, 4:-4]

    #input = (16,3,32,32)
    def forward(self, input):
        #NCHW
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft #fft包装器
        periodize = self.periodize #划分时期
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      64+8+1,
                      32,
                      32)               #new的这个是一个全趋于零的tensor，知识type和device都与inputs保持一致。
        
        #torch.Size([16, 3, 40, 40, 2])
        U_r = pad(input)#copies and view the real to complex
        
        #进行傅里叶变换
        #torch.Size([16, 3, 40, 40, 2])
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
        
        # First low pass filter（）
        #torch.Size([16, 3, 10, 10, 2])
        U_1_c = cdgmm(U_0_c, phi[0], jit=self.jit)                                                                                  #phi

        #torch.Size([16, 3, 10, 10])
        U_J_r = fft(U_1_c, 'C2R')#傅里叶反变换,将输入的最后一维去掉，即去掉2.
                 
        ##最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要 #任何就地改变一个tensor的操作都以_为后缀。例如：x.copy_(y), x.t_()，都会改变x。
        #unpad:输入([16, 3, 10, 10])，输出：([16, 3, 8, 8])
        #torch.Size([16, 3, 81, 8, 8])，这个81是1 + 8*J + 8*8*J*(J - 1) // 2。
        #这个操作实际上是把unpad(U_J_r)的值复制到S对应的位置上，详细见2.py
        S[..., n, :, :].copy_(unpad(U_J_r))

        n = n + 1#n=1,第2个channel

        #psi[n1]一共有16个，每一个都是一个字典，属性为j和theta,0三个，其中0的值就是滤波器（张量）
        #结束后n=81（(j-1)+...+1=64）
        for n1 in range(len(psi)):#len(psi)=16
            j1 = psi[n1]['j']
            #U_0_c:torch.Size([16, 3, 40, 40, 2]);psi[n1][0]:(40*40*2)
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)#进行小波变换后：torch.Size([16, 3, 40, 40, 2])
            if(j1 > 0):
                break
				#torch.Size([16, 3, 20, 20, 2]),尺寸除以2
                U_1_c = periodize(U_1_c, k=2 ** j1)
            fft(U_1_c, 'C2C', inverse=True, inplace=True)#傅里叶反变换
            U_1_c = fft(modulus(U_1_c), 'C2C')#取模再傅里叶变换（运行过了，这里取模运算之后还是复数(16*3*40*40*2)）

            # Second low pass filter
            # U_2_c = periodize(cdgmm(U_1_c, phi[j1], jit=self.jit), k=2**(J-j1))#torch.Size([16, 3, 10, 10, 2])
            U_2_c = cdgmm(U_1_c, phi[j1], jit=self.jit)                                                                               #phi
            U_J_r = fft(U_2_c, 'C2R')#torch.Size([16, 3, 10, 10])
            S[..., n, :, :].copy_(unpad(U_J_r))#unpad(U_J_r):torch.Size([16, 3, 8, 8])
            n = n + 1

            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    # U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1], jit=self.jit), k=2 ** (j2-j1))
                    U_2_c = cdgmm(U_1_c, psi[n2][j1], jit=self.jit)
                    fft(U_2_c, 'C2C', inverse=True, inplace=True)#傅里叶反变换
                    U_2_c = fft(modulus(U_2_c), 'C2C')#取模再傅里叶变换

                    # Third low pass filter
                    # U_2_c = periodize(cdgmm(U_2_c, phi[j2], jit=self.jit), k=2 ** (J-j2))
                    U_2_c = cdgmm(U_2_c, phi[0], jit=self.jit)                                                                        #phi
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n, :, :].copy_(unpad(U_J_r))
                    n = n + 1

        if (n != 64+8+1):
            raise (RuntimeError('this is a debug'))   
            
        return S#输出：torch.Size([16, 3, 17, 32, 32])

    ##该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, input):
        return self.forward(input)



class Scattering_Gabor9_level2(object):
    """Scattering module.

    Runs scattering on an input image in NCHW（N代表数量， C代表channel，H代表高度，W代表宽度） format

    Input args:
        M, N: input image size
        J: number of layers
        pre_pad: if set to True, module expect pre-padded images
        jit: compile kernels on the fly for speed（快速编译核）
    """
    #M = 32, N = 32, J = 2, pre_pad=False
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering_Gabor9_level2, self).__init__()
        print('Scattering_Gabor9_level2'.center(80, '='))
        self.M, self.N, self.J = M, N, J   
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit) #模数
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)#每个方向的填充长度=4

        # Create the filters
        #返回16个小波函数以及2个尺度函数（J=2，L=8）
        filters = filters_bank(self.M_padded, self.N_padded, J) #(40,40,2)

        #16个小波函数
        self.Psi = filters['psi']
        #2个尺度函数
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J  #40
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J  #40

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)#fill_(0)就是new一个真正全零的tensor。
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module(input)#每个方向的填充长度=4
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)#这里面的2应该就是从实数转为复数的意思
            output.narrow(4, 0, 1).copy_(out_.unsqueeze(4))#复制给第4个维度从0到1索引
        return output

    #最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要
    def _unpad(self, in_):        		
        return in_[..., 4:-4, 4:-4]

    #input = (16,3,32,32)
    def forward(self, input):
        #NCHW
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft #fft包装器
        periodize = self.periodize #划分时期
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      64+8+1,
                      32,
                      32)               #new的这个是一个全趋于零的tensor，知识type和device都与inputs保持一致。
        
        #torch.Size([16, 3, 40, 40, 2])
        U_r = pad(input)#copies and view the real to complex
        
        #进行傅里叶变换
        #torch.Size([16, 3, 40, 40, 2])
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
        
        # First low pass filter（）
        #torch.Size([16, 3, 10, 10, 2])
        U_1_c = cdgmm(U_0_c, phi[0], jit=self.jit)                                                                                  #phi

        #torch.Size([16, 3, 10, 10])
        U_J_r = fft(U_1_c, 'C2R')#傅里叶反变换,将输入的最后一维去掉，即去掉2.
                 
        ##最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要 #任何就地改变一个tensor的操作都以_为后缀。例如：x.copy_(y), x.t_()，都会改变x。
        #unpad:输入([16, 3, 10, 10])，输出：([16, 3, 8, 8])
        #torch.Size([16, 3, 81, 8, 8])，这个81是1 + 8*J + 8*8*J*(J - 1) // 2。
        #这个操作实际上是把unpad(U_J_r)的值复制到S对应的位置上，详细见2.py
        S[..., n, :, :].copy_(unpad(U_J_r))

        n = n + 1#n=1,第2个channel

        #psi[n1]一共有16个，每一个都是一个字典，属性为j和theta,0三个，其中0的值就是滤波器（张量）
        #结束后n=81（(j-1)+...+1=64）
        for n1 in range(len(psi)):#len(psi)=16
            j1 = psi[n1]['j']
            #U_0_c:torch.Size([16, 3, 40, 40, 2]);psi[n1][0]:(40*40*2)
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)#进行小波变换后：torch.Size([16, 3, 40, 40, 2])
            if(j1 > 0):
                break
				#torch.Size([16, 3, 20, 20, 2]),尺寸除以2
                U_1_c = periodize(U_1_c, k=2 ** j1)

            U_J_r = fft(U_1_c, 'C2R')#torch.Size([16, 3, 10, 10])
            S[..., n, :, :].copy_(unpad(U_J_r))#unpad(U_J_r):torch.Size([16, 3, 8, 8])
            n = n + 1

            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = cdgmm(U_1_c, psi[n2][j1], jit=self.jit)
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n, :, :].copy_(unpad(U_J_r))
                    n = n + 1

        if (n != 64+8+1):
            raise (RuntimeError('this is a debug'))   
            
        return S#输出：torch.Size([16, 3, 17, 32, 32])

    ##该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, input):
        return self.forward(input)


class Scattering_shearlet_level2(object):
    """Scattering module.

    Runs scattering on an input image in NCHW（N代表数量， C代表channel，H代表高度，W代表宽度） format

    Input args:
        M, N: input image size
        J: number of layers
        pre_pad: if set to True, module expect pre-padded images
        jit: compile kernels on the fly for speed（快速编译核）
    """
    #M = 32, N = 32, J = 2, pre_pad=False
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering_shearlet_level2, self).__init__()
        print('Scattering_shearlet_level2'.center(80, '='))
        self.M, self.N, self.J = M, N, J   
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit) #模数
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)#每个方向的填充长度=4

        # Create the filters
        #返回18个小波函数以及2个尺度函数（J=2，L=9）
        filters = filters_bank(self.M_padded, self.N_padded, J) #(40,40,2)

        #18个小波函数
        self.Psi = filters['psi']
        #2个尺度函数
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J  #40
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J  #40

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)#fill_(0)就是new一个真正全零的tensor。
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module(input)#每个方向的填充长度=4
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)#这里面的2应该就是从实数转为复数的意思
            output.narrow(4, 0, 1).copy_(out_.unsqueeze(4))#复制给第4个维度从0到1索引
        return output

    #最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要
    def _unpad(self, in_):        		
        return in_[..., 4:-4, 4:-4]

    #input = (16,3,32,32)
    def forward(self, input):
        #NCHW
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft #fft包装器
        periodize = self.periodize #划分时期
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      81+9+1,
                      32,
                      32)               #new的这个是一个全趋于零的tensor，知识type和device都与inputs保持一致。
        
        #torch.Size([16, 3, 40, 40, 2])
        U_r = pad(input)#copies and view the real to complex
        
        #进行傅里叶变换
        #torch.Size([16, 3, 40, 40, 2])
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
        
        # First low pass filter（）
        #torch.Size([16, 3, 10, 10, 2])
        U_1_c = cdgmm(U_0_c, phi[0], jit=self.jit)                                                                                  #phi

        #torch.Size([16, 3, 10, 10])
        U_J_r = fft(U_1_c, 'C2R')#傅里叶反变换,将输入的最后一维去掉，即去掉2.
                 
        ##最后两个维度的第0个元素和最后一个元素不要，即最后32*32的图像中外面一圈不要 #任何就地改变一个tensor的操作都以_为后缀。例如：x.copy_(y), x.t_()，都会改变x。
        #unpad:输入([16, 3, 10, 10])，输出：([16, 3, 8, 8])
        #torch.Size([16, 3, 81, 8, 8])，这个81是1 + 8*J + 8*8*J*(J - 1) // 2。
        #这个操作实际上是把unpad(U_J_r)的值复制到S对应的位置上，详细见2.py
        S[..., n, :, :].copy_(unpad(U_J_r))

        n = n + 1#n=1,第2个channel

        #psi[n1]一共有16个，每一个都是一个字典，属性为j和theta,0三个，其中0的值就是滤波器（张量）
        #结束后n=81（(j-1)+...+1=64）
        for n1 in range(len(psi)):#len(psi)=16
            j1 = psi[n1]['j']
            #U_0_c:torch.Size([16, 3, 40, 40, 2]);psi[n1][0]:(40*40*2)
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)#进行小波变换后：torch.Size([16, 3, 40, 40, 2])
            if(j1 > 0):
                break
				#torch.Size([16, 3, 20, 20, 2]),尺寸除以2
                U_1_c = periodize(U_1_c, k=2 ** j1)

            U_J_r = fft(U_1_c, 'C2R')#torch.Size([16, 3, 10, 10])
            S[..., n, :, :].copy_(unpad(U_J_r))#unpad(U_J_r):torch.Size([16, 3, 8, 8])
            n = n + 1

            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = cdgmm(U_1_c, psi[n2][j1], jit=self.jit)
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n, :, :].copy_(unpad(U_J_r))
                    n = n + 1

        if (n != 81+9+1):
            raise (RuntimeError('this is a debug'))   
            
        return S

    ##该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, input):
        return self.forward(input)



if __name__ == '__main__':
    imgs = np.zeros((2,3,32,32)) #python的zeros要带括号                     #这里改一共有多少张图像
    for i in range(1):
        # img = cv2.imread('D:/1Peostgraduate/frog.png') #(32,32,3)
        img = cv2.imread('D:/1Peostgraduate/zoneplate_huatu_resize.png')
        img = img.transpose(2, 0, 1).astype(np.float32) #(3,32,32)
        imgs[i,:,:,:] = img
    img = cv2.imread('D:/1Peostgraduate/30.png') #(32,32,3)
    img = img.transpose(2, 0, 1).astype(np.float32) #(3,32,32)
    imgs[1,:,:,:] = img                                                 #现在改成只有2张图像

    #转换成torch格式
    imgs = torch.from_numpy(imgs).cuda()
    imgs = imgs.type(torch.cuda.FloatTensor)      

    scat = Scattering_Gabor9_level2(M=32, N=32, J=2, pre_pad=False).cuda()     #该散射网络
    inputs = scat(imgs)

    #转换回numpy格式，显示图像(必须是40,40,3)的格式//但是要注意这里和python的数据风格不一样，python还是切片在前面
    inputs = inputs.cpu().numpy() #(16, 3, 13, 40, 40) 
    inputs = inputs.transpose(0,2,3,4,1) #(16,13,40,40,3)
    print(inputs.shape)

    #保存图像
    path = 'Scattering_Gabor9_level2'
    if not os.path.exists(path):
        os.makedirs(path)    
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            # cv2.imwrite('show/aaa/'+str(i)+'_'+str(j)+'.png',inputs[i,j,:,:,:])
            io.imsave(path + '/' +str(i) + '_' + str(j) + '.png', inputs[i,j,:,:,:])
        
        
