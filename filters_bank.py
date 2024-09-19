"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['filters_bank']

import torch
import numpy as np
import scipy.fftpack as fft
from read_mat import read_mat_compex
import scipy.io

#40,40,2
#小波函数psi和尺度函数phi,对应希腊字母psiψ，phiφ
#尺度函数与小波函数共同构造了信号的分解。这里尺度函数可以由低通滤波器构造，而小波函数则由高通滤波器实现。
def filters_bank(M, N, J, L=8):
    filters = {}
    filters['psi'] = []

###################################################################Gabor小波####################################################################
    # offset_unpad = 0
    # #psi有J*L=16个
    # for j in range(J):
    #     for theta in range(L):
    #         psi = {}    #字典
    #         psi['j'] = j
    #         psi['theta'] = theta
    #         #生成一个小波，##返回40*40的复数矩阵(每一个元素都是复数)
    #         psi_signal = morlet_2d(M, N, 0.8 * 2**j, (int(L-L/2-1)-theta) * np.pi / L, 3.0 / 4.0 * np.pi /2**j,offset=offset_unpad) # The 5 is here just to match the LUA implementation :)
    #         psi_signal_fourier = fft.fft2(psi_signal)#对小波进行傅里叶变换(返回40*40的复数矩阵)
    #         #psi[0]=40*40*2;psi[1]=20*20*2
    #         for res in range(j + 1):#res = 0，1
    #             psi_signal_fourier_res = crop_freq(psi_signal_fourier, res)#返回的是(M // 2 ** res, N // 2 ** res)的复数矩阵
    #             #np.real：实部；np.imag：虚部，这里的意思应该是...返回(M // 2 ** res)*(N // 2 ** res)*2
    #             psi[res]=torch.FloatTensor(np.stack((np.real(psi_signal_fourier_res), np.imag(psi_signal_fourier_res)), axis=2))
    #             # Normalization to avoid doing it with the FFT!
    #             psi[res].div_(M*N// 2**(2*j))
    #         filters['psi'].append(psi)


################################################################matlab剪切波######################################################################
    print('using shearlet filter'.center(80,'='))
    offset_unpad = 0
    #psi有J*L=16个
    L=9
    for j in range(J):
        for theta in range(L):
            psi = {}    #字典
            psi['j'] = j
            psi['theta'] = theta

            #剪切波滤波器
            data = scipy.io.loadmat('data_1.mat') #264和1
            psi_signal_fourier =  read_mat_compex(data)[theta,:,:]

            #psi[0]=40*40*2;psi[1]=20*20*2
            for res in range(j + 1):#res = 0，1
                psi_signal_fourier_res = crop_freq(psi_signal_fourier, res)#返回的是(M // 2 ** res, N // 2 ** res)的复数矩阵
                #np.real：实部；np.imag：虚部，这里的意思应该是...返回(M // 2 ** res)*(N // 2 ** res)*2
                psi[res]=torch.FloatTensor(np.stack((np.real(psi_signal_fourier_res), np.imag(psi_signal_fourier_res)), axis=2))
                # Normalization to avoid doing it with the FFT!
                psi[res].div_(M*N// 2**(2*j))
            filters['psi'].append(psi)


####################################################################python剪切波##################################################################
    # offset_unpad = 0
    # L=13
    # #python的剪切波
    # Psi = scalesShearsAndSpectra((40,40), numOfScales=None, 
                     # realCoefficients=True)
    # psi_signal_fourier_bank = fft.fft2(Psi) 
    
    # for j in range(J):
        # for theta in range(L):
            # psi = {}    #字典
            # psi['j'] = j
            # psi['theta'] = theta
            # # #生成一个小波，##返回40*40的复数矩阵(每一个元素都是复数)
            # # psi_signal = morlet_2d(M, N, 0.8 * 2**j, (int(L-L/2-1)-theta) * np.pi / L, 3.0 / 4.0 * np.pi /2**j,offset=offset_unpad) # The 5 is here just to match the LUA implementation :)
            # # psi_signal_fourier = fft.fft2(psi_signal)#对小波进行傅里叶变换(返回40*40的复数矩阵)
            
            # psi_signal_fourier = psi_signal_fourier_bank[:,:,theta]
            
            # #psi[0]=40*40*2;psi[1]=20*20*2
            # for res in range(j + 1):#res = 0，1
                # psi_signal_fourier_res = crop_freq(psi_signal_fourier, res)#返回的是(M // 2 ** res, N // 2 ** res)的复数矩阵
                # #np.real：实部；np.imag：虚部，这里的意思应该是...返回(M // 2 ** res)*(N // 2 ** res)*2
                # psi[res]=torch.FloatTensor(np.stack((np.real(psi_signal_fourier_res), np.imag(psi_signal_fourier_res)), axis=2))
                # # Normalization to avoid doing it with the FFT!
                # psi[res].div_(M*N// 2**(2*j))
            # filters['psi'].append(psi)






                        
    #phi有1个,phi[0]=40*40*2;phi[1]=20*20*2
    filters['phi'] = {}
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0, offset=offset_unpad)#Gabor变换
    phi_signal_fourier = fft.fft2(phi_signal)
    filters['phi']['j'] = J
    for res in range(J):
        phi_signal_fourier_res = crop_freq(phi_signal_fourier, res)
        filters['phi'][res]=torch.FloatTensor(np.stack((np.real(phi_signal_fourier_res), np.imag(phi_signal_fourier_res)), axis=2))
        filters['phi'][res].div_(M*N // 2 ** (2 * J))

    # print(filters['phi'])
    return filters#返回16个小波函数以及2个尺度函数（J=2，L=8）


#我不知道在干嘛，返回的是(M // 2 ** res, N // 2 ** res)的复数矩阵
def crop_freq(x, res):
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), np.complex64)

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)#将傅里叶变换的结果只保留四个角，其余全是0

    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

    return crop


#M=40, N=40, sigma=(0.8,1.6), theta=(3,2,1,0,-1,-2,-3,-4)*8/pi, xi=3*pi/4或3*pi/8, offset=0
def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=None):
    """ This function generated a morlet：这个函数生成一个小波"""
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)#返回40*40的复数矩阵
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor#返回40*40的复数矩阵

#为解决傅氏变换局部频率变化的不足， 而在其基础上增加窗函数， 实现有效获得信号的局部信息， 因此Gabor变换是一种基于窗口的短时傅氏变换。
def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
    gab = np.zeros((M, N), np.complex64)#40*40的复数矩阵
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab = gab + np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab = gab / norm_factor

    #用于将FFT变换之后的频谱显示范围从[0, N]变为：[-N/2, N/2-1](N为偶数)或者[-(N-1)/2, (N-1)/2](N为奇数)。
    #注意这里fft_shift=None，所以实际上并没有运行
    if (fft_shift):
        gab = np.fft.fftshift(gab, axes=(0, 1))
    return gab##返回40*40的复数矩阵
