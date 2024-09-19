#集群   nthread, gpu, WRN12-8, filterbanks=shearlet, Scattering_shearlet_level2, save
#sampleSize=100, frequency_save=5000, mul=100, batchsize=16, self.nfscat = 91, attention=None, frequency_print=70


import os
import re
import json
import numpy as np
  
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import torch
  
from torchnet import dataset, meter
from torchnet.engine import Engine
  
from torch.autograd import Variable
from torch.nn import Parameter
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision.datasets as datasets

import cvtransforms
from scattering import Scattering_shearlet_py, Scattering_Gabor2_9, Scattering_Gabor9, Scattering_Gabor9_level2, Scattering_no_downsample, Scattering_shearlet_level2

import torchnet as tnt
from Scatter_WRN_1 import resnet12_8_scat, just_fc
# from Sattering_WRN_attention import resnet12_8_scat, just_fc                                        
import cv2
import argparse



def parse():
    parser = argparse.ArgumentParser(description='Scattering on CIFAR')
    # Model options
    parser.add_argument('--nthread', default=8, type=int)
  
  
    # Training options
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--sampleSize', default=100, type=int)
    parser.add_argument('--mul', default=100, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--batchSize', default=16, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weightDecay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--resume', default='', type=str)
  
    # Save options
    parser.add_argument('--base_save', default='shearlet_2levels_WRN12-8_100_BS32_seed=', type=str,
                        help='save parameters and logs in this folder')
    parser.add_argument('--save', default='', type=str,
                        help='save parameters and logs in this folder')
    parser.add_argument('--frequency_save', default=5000,
                        type=int,
                        help='Frequency at which one should save')
    parser.add_argument('--frequency_test', default=25,
                        type=int,
                        help='Frequency at which one should save')
  
    parser.add_argument('--ngpu', default=2, type=int,
                        help='number of GPUs to use for training')
    parser.add_argument('--gpu_id', default='0,1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--scat', default=2, type=int,
                        help='scattering scale, j=0 means no scattering')
    parser.add_argument('--N', default=32, type=int,
                        help='size of the crop')
    parser.add_argument('--randomcrop_pad', default=4, type=float)
  
  
    # Display options
    parser.add_argument('--frequency_print', default=70,
                        type=int,
                        help='Frequency at which one should save')
    return parser
  
  
cudnn.benchmark = True
 
parser = parse()    #参数
opt = parser.parse_args()
 
  
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id #选用第0个显卡
torch.randn(8).cuda()
epoch_step = list(np.array(json.loads(opt.epoch_step))*opt.mul)    #[60,120,160]*20
opt.epochs=opt.epochs*opt.mul

print('parsed options:', vars(opt))
data_time = 1

print('epoch_step',epoch_step)
def create_dataset(opt, mode):
    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),  #转换数据类型
        lambda x: x / 255.0,
        # cvtransforms.Normalize([125.3, 123.0, 113.9], [63.0,  62.1,  66.7]),
        lambda x: x.transpose(2, 0, 1).astype(np.float32),
        torch.from_numpy,
    ])
  
    #翻转、填充、裁剪
    train_transform = tnt.transform.compose([
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.Pad(opt.randomcrop_pad, cv2.BORDER_REFLECT),
        cvtransforms.RandomCrop(32),
        convert,
    ])
     
    #opt.dataset：CIFAR10
    #getattr() 函数用于返回一个对象属性值, train=mode 获取相应的数据集
    ds = getattr(datasets, opt.dataset)('../../Scatting', train=mode, download=True)#下载数据集
    smode = 'train' if mode else 'test'
    if mode:
        from numpy.random import RandomState #伪随机数生成器
        prng = RandomState(opt.seed)
         
        assert(opt.sampleSize%10==0)
         
        random_permute=prng.permutation(np.arange(0,5000))[0:int(opt.sampleSize/10)]

        labels = np.array(getattr(ds,'targets'))
        data = getattr(ds,'data')
       
        classes=np.unique(labels)
        inds_all=np.array([],dtype='int32')
        for cl in classes:
            inds=np.where(np.array(labels)==cl)[0][random_permute]
            inds_all=np.r_[inds,inds_all]    #np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。

        ds = tnt.dataset.TensorDataset([
            data[inds_all,:],#.transpose(0, 2, 3, 1),
            labels[inds_all].tolist()])
    else:
        ds = tnt.dataset.TensorDataset([
            getattr(ds, 'data'),#.transpose(0, 2, 3, 1),
            getattr(ds, 'targets')])
    return ds.transform({0: train_transform if mode else convert})
  
def main():

    for seed_idx in range(1,11,2):

        opt.seed = seed_idx
        opt.save = opt.base_save + str(seed_idx)
        print('seed={}'.format(seed_idx).center(80,'='))    

        if not os.path.exists(opt.save):
            os.mkdir(opt.save)
    
        if opt.scat>0:
            #__dict__查看对象内部所有属性名和属性值组成的字典
            #opt.model = 'resnet12_8_scat'
            model, params, stats = resnet12_8_scat(N=opt.N,J=opt.scat)#N = 32，J = 2                                               #在这里更改ResNet or FC
        else:
            model, params, stats = models.__dict__[opt.model]()
        
        def create_optimizer(opt, lr):
            print('creating optimizer with lr = %f'% lr)
            return torch.optim.SGD(params.values(), lr, opt.momentum, weight_decay=opt.weightDecay)
        def get_iterator(mode):#mode 测试或者训练（测试不打乱）
            ds = create_dataset(opt, mode)#（测试：返回测试数据集）
            return ds.parallel(batch_size=opt.batchSize, shuffle=mode,
                            num_workers=opt.nthread, pin_memory=False)
    
        optimizer = create_optimizer(opt, opt.lr)
    
        #迭代器的元素是(16,3,32,32),(16,1)，分别代表16个图像以及标签
        iter_test = get_iterator(False)#创建测试数据集的迭代器
        iter_train = get_iterator(True)
        
    
        #散射变换
        if opt.scat>0:
            scat = Scattering_shearlet_level2(M=opt.N, N=opt.N, J=opt.scat, pre_pad=False).cuda()                             #在这里更改散射网络模型
    
        epoch = 0
        if opt.resume != '':
            resumeFile=opt.resume
            if not resumeFile.endswith('pt7'):
                resumeFile=torch.load(opt.resume + '/latest.pt7')['latest_file']
                state_dict = torch.load(resumeFile)
                epoch = state_dict['epoch']
                params_tensors, stats = state_dict['params'], state_dict['stats']
                for k, v in params.items():
                    v.data.copy_(params_tensors[k])
                optimizer.load_state_dict(state_dict['optimizer'])
                print('model was restored from epoch:',epoch)
    
        print('\nParameters:')
        print(pd.DataFrame([(key, v.size(), torch.typename(v.data)) for key, v in params.items()]))
        print('\nAdditional buffers:')
        print(pd.DataFrame([(key, v.size(), torch.typename(v)) for key, v in stats.items()]))
        n_parameters = sum([p.numel() for p in list(params.values()) + list(stats.values())])
        print('\nTotal number of parameters: %f'% n_parameters)
    
        meter_loss = meter.AverageValueMeter()#测量并返回添加到其中的任何数字集合的平均值和标准差，测量一组例子的平均损失是很有用的
        classacc = meter.ClassErrorMeter(topk=[1, 5], accuracy=False)#用于统计分类误差，topk指定分别统计top1和top5误差，返回错误率。
        timer_data = meter.TimeMeter('s')#用于统计events之间的时间
        timer_sample = meter.TimeMeter('s')
        timer_train = meter.TimeMeter('s')
        timer_test = meter.TimeMeter('s')
    
    
        def h(sample):
            inputs = sample[0].cuda()#sample[0]代表一个batch的图像。
            if opt.scat > 0:
                inputs = scat(inputs)#散射变换，返回散射系数S：torch.Size([16, 3, 81, 8, 8])
            targets = Variable(sample[1].cuda().long())#真实标签
            
            #Scatter_WEN中的resnet12_8_scat
            if sample[2]:
                model.train()
            else:
                model.eval()
            #inputs：（batch_size,3,81,8,8）
            y = torch.nn.parallel.data_parallel(model, inputs, np.arange(opt.ngpu).tolist())#将散射变换的结果送入resnet12_8_scat模型进行训练
            #      损失函数
            return F.cross_entropy(y, targets), y
    
    
        def log(t, state):
            if(t['epoch']>0 and t['epoch']%opt.frequency_save==0):
                torch.save(dict(params={k: v.data.cpu() for k, v in params.items()},
                            stats=stats,
                            optimizer=state['optimizer'].state_dict(),
                            epoch=t['epoch']),
                    open(os.path.join(opt.save, 'epoch_%i_model.pt7' % t['epoch']), 'wb'))
                torch.save( dict(latest_file=os.path.join(opt.save, 'epoch_%i_model.pt7' % t['epoch'])
                                ),
                            open(os.path.join(opt.save, 'latest.pt7'), 'wb'))
    
            z = vars(opt).copy()
            z.update(t)
            logname = os.path.join(opt.save, 'log.txt')
            with open(logname, 'a') as f:
                f.write('json_stats: ' + json.dumps(z) + '\n')
            print(z)
    
        def on_sample(state):
            global data_time
            data_time = timer_data.value()#value() 返回从reset()到现在的时间消耗
            timer_sample.reset()#reset() 重置timer，unit counter
            state['sample'].append(state['train'])
    
    
        def on_forward(state):
            prev_sum5=classacc.sum[5]#classacc：用于统计分类误差，topk指定分别统计top1和top5误差，返回错误率。
            prev_sum1 = classacc.sum[1]
            classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
            meter_loss.add(state['loss'].item())
    
            next_sum5 = classacc.sum[5]
            next_sum1 = classacc.sum[1]
            n =  state['output'].data.size(0)
            curr_top5=100.0*(next_sum5-prev_sum5)/n        #top5错误率（用当前的错误累加-前一轮的错误累加=当前轮次的错误率）
            curr_top1 = 100.0*(next_sum1 - prev_sum1) / n  #top1错误率
            sample_time = timer_sample.value()
            timer_data.reset()
            if(state['train']):
                txt = 'Train:'
            else:
                txt = 'Test'
            if(state['t']%opt.frequency_print==0 and state['t']>0):
                print('%s [%i,%i/%i] ; loss: %.3f (%.3f) ; acc5: %.2f (%.2f) ; acc1: %.2f (%.2f) ; data %.3f ; time %.3f' %
                    (txt, state['epoch'],state['t']%len(state['iterator']),
                    len(state['iterator']),
                    state['loss'].item(),
                    meter_loss.value()[0],
                    100-curr_top5,
                    100-classacc.value(5),
                    100-curr_top1,
                    100-classacc.value(1),
                    data_time,
                    sample_time
                    ))
    
    
        def on_start(state):
            state['epoch'] = epoch
    
        def on_start_epoch(state):
            classacc.reset()
            meter_loss.reset()
            timer_train.reset()
    

            state['iterator'] = iter_train
    
            epoch = state['epoch'] + 1
            if epoch in epoch_step:
                print('changing LR')
                lr = state['optimizer'].param_groups[0]['lr']
                state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)
    
        #注意！只有在一个epoch结束的时候，才会检测state['t']%opt.frequency_test==0。
        def on_end_epoch(state):
            if(state['epoch']%opt.frequency_test==0 and state['t']>0):
                train_loss = meter_loss.value()
                train_acc = classacc.value()
                train_time = timer_train.value()
                meter_loss.reset()
                classacc.reset()
                timer_test.reset()
        
                engine.test(h, iter_test)    #这里是每个epoch最后，所以是测试，下面最后一行才是训练
        
                log({
                    "train_loss": train_loss[0],
                    "train_acc": 100-train_acc[0],
                    "test_loss": meter_loss.value()[0],
                    "test_acc": 100-classacc.value()[0],
                    "epoch": state['epoch'],
                    "n_parameters": n_parameters,
                    "train_time": train_time,
                    "test_time": timer_test.value(),
                }, state)
    
    
    
        #它将训练过程和测试过程进行包装，抽象成一个类，提供train和test方法和一个hooks.
        #hooks包括on_start, on_sample, on_forward, on_update,  on_end_epoch,  on_end，可以自己制定函数，
        #在开始，load数据，forward，更新还有epoch结束以及训练结束时执行。一般是用开查看和保存模型训练过程的一些结果。
        engine = Engine()
        engine.hooks['on_sample'] = on_sample#每次采样一个样本之后的操作
        engine.hooks['on_forward'] = on_forward#在model:forward()之后的操作
        engine.hooks['on_start_epoch'] = on_start_epoch#每一个epoch前的操作
        engine.hooks['on_end_epoch'] = on_end_epoch#每一个epoch结束时的操作
        engine.hooks['on_start'] = on_start#用于训练开始前的设置和初始化
        engine.train(h, iter_train, opt.epochs, optimizer)#在数据集上训练数据

'''
外部可以通过state变量与Engine训练过程交互
state = {
['network'] = network, --设置了model
['criterion'] = criterion, -- 设置损失函数
['iterator'] = iterator, -- 数据迭代器
['lr'] = lr, -- 学习率
['lrcriterion'] = lrcriterion, --
['maxepoch'] = maxepoch, --最大epoch数
['sample'] = {}, -- 当前采集的样本，可以在onSample中通过该阈值查看采样样本
['epoch'] = 0 , -- 当前的epoch
['t'] = 0, -- state['t']实际上是当前batch数。
['training'] = true -- 训练过程
}
'''  
  
if __name__ == '__main__':
    main()
