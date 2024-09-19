"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""
from collections import defaultdict, namedtuple

import torch
from skcuda import cublas, cufft
from pynvrtc.compiler import Program
import numpy as np
from cupy.cuda.function import Module
from cupy.cuda import device
from string import Template


Stream = namedtuple('Stream', ['ptr'])


def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


def get_compute_arch(t):
    return 'compute_%s' % device.Device().compute_capability


def iscomplex(input):
    return input.size(-1) == 2



#输入的一个例子torch.Size([16, 3, 40, 40, 2])
#池化，按k缩小输入数据尺寸
class Periodize(object):
    """This class builds a wrapper to the periodiziation kernels and cache them.
        这个类为periodiziation内核构建一个包装器并缓存它们。
        """
    def __init__(self, jit=True):
        self.periodize_cache = defaultdict(lambda: None) #这种方式可以为一个字典对象不存在的key自动给出一个默认的value。
        self.block = (32, 32, 1)
        self.jit = jit

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, input, k):
        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2) #缩小k倍，k为2的尺度次方  #随机的

        if not self.jit or isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
            y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)

            #mean：在该维度上求平均，同时会降一维。（对3、5维求平均）
            #squeeze：把这一维度（如果是单维度）删掉。
            #这一操作相当于尺度为k的平均池化（尺寸减小k倍）
            out = y.mean(4).squeeze(4).mean(2).squeeze(2)
            return out

        if not iscomplex(input):
            raise (TypeError('The input and outputs should be complex'))

        #contiguous:将input内存地址连续化，因为torch.view等方法操作需要连续的Tensor。
        input = input.contiguous()
       
        if (self.periodize_cache[(input.size(), out.size(), input.get_device())] is None):
            # 这一部分是实现功能的主要代码，${}中括号内的内容应该就是替代字符
            kernel = '''
            #define NW ${W} / ${k}
            #define NH ${H} / ${k}
            extern "C"
            __global__ void periodize(const ${Dtype}2 *input, ${Dtype}2 *output)
            {
              int tx = blockIdx.x * blockDim.x + threadIdx.x;
              int ty = blockIdx.y * blockDim.y + threadIdx.y;
              int tz = blockIdx.z * blockDim.z + threadIdx.z;
              if(tx >= NW || ty >= NH || tz >= ${B})
                return;
              input += tz * ${H} * ${W} + ty * ${W} + tx;
              ${Dtype}2 res = make_${Dtype}2(0.f, 0.f);
              for (int j=0; j<${k}; ++j)
                for (int i=0; i<${k}; ++i)
                {
                  const ${Dtype}2 &c = input[j * NH * ${W} + i * NW];
                  res.x += c.x;
                  res.y += c.y;
                }
              res.x /= ${k} * ${k};
              res.y /= ${k} * ${k};
              output[tz * NH * NW + ty * NW + tx] = res;
            }
            '''
            B = input.nelement() // (2*input.size(-2) * input.size(-3)) ##pytorch中的 nelement() 可以统计 tensor (张量) 的元素的个数。
            W = input.size(-2)
            H = input.size(-3)
            k = input.size(-2) // out.size(-2) #缩小k倍，k为2的尺度次方

            #Template有两个substitute方法。用substitute时。所带的keywords必须被替代字符串配对，不然会抛出ValueError异常
            #Template为python string提供的一个字符串模板功能。主要用于文本处理。
            kernel = Template(kernel).substitute(B=B, H=H, W=W, k=k, Dtype=getDtype(input))
            name = str(input.get_device())+'-'+str(B)+'-'+str(k)+'-'+str(H)+'-'+str(W)+'-periodize.cu'

            #使用 pynvrtc 提供的高层接口来编译上面定义的 CUDA 代码
            prog = Program(kernel, name)
            #将 CUDA 源码编译为 PTX （GPU上的汇编语言）。这实际运行中，GPU 驱动会负责将 PTX 翻译为机器码进行执行。
            ptx = prog.compile(['-arch='+get_compute_arch(input)])

            #为了方便在 python 程序中直接调用，我们需要将 PTX 函数进行封装。这个可以借助 cupy 方便的实现。
            module = Module()
            module.load(bytes(ptx.encode()))

            self.periodize_cache[(input.size(), out.size(), input.get_device())] = module

        grid = (self.GET_BLOCKS(out.size(-3), self.block[0]),
                self.GET_BLOCKS(out.size(-2), self.block[1]),
                self.GET_BLOCKS(out.nelement() // (2*out.size(-2) * out.size(-3)), self.block[2]))
        periodize = self.periodize_cache[(input.size(), out.size(), input.get_device())].get_function('periodize')
        periodize(grid=grid, block=self.block, args=[input.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


#取模
class Modulus(object):
    """This class builds a wrapper to the moduli kernels and cache them.
        """
    def __init__(self, jit=True):
        self.modulus_cache = defaultdict(lambda: None)
        self.CUDA_NUM_THREADS = 1024
        self.jit = jit

    def GET_BLOCKS(self, N):
        return (N + self.CUDA_NUM_THREADS - 1) // self.CUDA_NUM_THREADS

    def __call__(self, input):
        if not self.jit or not isinstance(input, torch.cuda.FloatTensor):
            norm = input.norm(2, input.dim() - 1)
            return torch.stack([norm, norm.new(norm.size()).zero_()], -1)

        out = input.new(input.size())
        ##contiguous:将input内存地址连续化，因为torch.view等方法操作需要连续的Tensor。
        input = input.contiguous()

        if not iscomplex(input):
            raise TypeError('The input and outputs should be complex')

        if (self.modulus_cache[input.get_device()] is None):
            kernel = """
            extern "C"
            __global__ void abs_complex_value(const float * x, float2 * z, int n)
            {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            z[i] = make_float2(normf(2, x + 2*i), 0);

            }
            """
            print('modulus.cu')
            prog = Program(kernel, 'modulus.cu')
            ptx = prog.compile(['-arch='+get_compute_arch(input)])
            module = Module()
            module.load(bytes(ptx.encode()))
            self.modulus_cache[input.get_device()] = module
        fabs = self.modulus_cache[input.get_device()].get_function('abs_complex_value')
        fabs(grid=(self.GET_BLOCKS(int(out.nelement())//2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[input.data_ptr(), out.data_ptr(), out.numel() // 2],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


class Fft(object):
    """This class builds a wrapper（包装器） to the FFTs kernels and cache them.

    As a try, the library will purely work with complex data. The FFTS are UNORMALIZED.
    作为一种尝试，这个库将只处理复杂的数据。FFT是非标准化的。
        """

    def __init__(self):
        self.fft_cache = defaultdict(lambda: None)

    def buildCache(self, input, type):
        k = input.ndimension() - 3
        n = np.asarray([input.size(k), input.size(k+1)], np.int32)
        batch = input.nelement() // (2*input.size(k) * input.size(k + 1))
        idist = input.size(k) * input.size(k + 1)
        istride = 1
        ostride = istride
        odist = idist
        rank = 2
        plan = cufft.cufftPlanMany(rank, n.ctypes.data, n.ctypes.data, istride,
                                   idist, n.ctypes.data, ostride, odist, type, batch)
        self.fft_cache[(input.size(), type, input.get_device())] = plan

    #当删除一个对象时，Python解释器也会默认调用一个方法，这个方法为__del__()方法。
    def __del__(self):
        for keys in self.fft_cache:
            try:
                cufft.cufftDestroy(self.fft_cache[keys])
            except:
                pass

    #__call__：该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, input, direction='C2C', inplace=False, inverse=False):
        #'C2R'：复数转实数
        if direction == 'C2R':
            inverse = True #inverse=True:傅里叶反变换，inverse=Flase:傅里叶正变换

        if not isinstance(input, torch.cuda.FloatTensor):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:
                input_np = input[..., 0].numpy() + 1.0j * input[..., 1].numpy()
                f = lambda x: np.stack((np.real(x), np.imag(x)), axis=len(x.shape))
                out_type = input.numpy().dtype

                if direction == 'C2R':
                    out = np.real(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    return torch.from_numpy(out)

                if inplace:
                    if inverse:
                        out = f(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    else:
                        out = f(np.fft.fft2(input_np)).astype(out_type)
                    input.copy_(torch.from_numpy(out))
                    return
                else:
                    if inverse:
                        out = f(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    else:
                        out = f(np.fft.fft2(input_np)).astype(out_type)
                    return torch.from_numpy(out)

        if not iscomplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        #'C2R'：复数转实数
        if direction == 'C2R':
            output = input.new(input.size()[:-1])
            if(self.fft_cache[(input.size(), cufft.CUFFT_C2R, input.get_device())] is None):
                self.buildCache(input, cufft.CUFFT_C2R)
            cufft.cufftExecC2R(self.fft_cache[(input.size(), cufft.CUFFT_C2R, input.get_device())],
                               input.data_ptr(), output.data_ptr())
            return output
        #'C2C'：复数转复数
        elif direction == 'C2C':
            output = input.new(input.size()) if not inplace else input
            flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
            if (self.fft_cache[(input.size(), cufft.CUFFT_C2C, input.get_device())] is None):
                self.buildCache(input, cufft.CUFFT_C2C)
            cufft.cufftExecC2C(self.fft_cache[(input.size(), cufft.CUFFT_C2C, input.get_device())],
                               input.data_ptr(), output.data_ptr(), flag)#flag确定进行正变换还是反变换
            return output


#把两个矩阵相乘
def cdgmm(A, B, jit=True, inplace=False):
    """This function uses the C-wrapper to use cuBLAS.(cuBLAS利用GPU加速向量、矩阵的线性运算。)
        """
    A, B = A.contiguous(), B.contiguous()

    if A.size()[-3:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 3:
        raise RuntimeError('The filters must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    if not jit or isinstance(A, (torch.FloatTensor, torch.DoubleTensor)):
        C = A.new(A.size())

        A_r = A[..., 0]#复数的实部（16，3，40，40，1）
        A_i = A[..., 1]#复数的虚部

        B_r = B[..., 0].unsqueeze(0)#在第0维增加一个维度：（1，40，40，1）
        B_i = B[..., 1].unsqueeze(0)

        C[..., 0].copy_(A_r * B_r - A_i * B_i)
        C[..., 1].copy_(A_r * B_i + A_i * B_r)

        # faster if B is actually real
        #B[...,1] = B[...,0]
        #C = A * B.unsqueeze(0).expand_as(A)
        return C if not inplace else A.copy_(C)
    else:
        C = A.new(A.size()) if not inplace else A
        m, n = B.nelement() // 2, A.nelement() // B.nelement() #pytorch中的 nelement() 可以统计 tensor (张量) 的元素的个数。
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        return C
