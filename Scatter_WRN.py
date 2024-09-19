import torch.nn as nn
import math
import torch.nn.functional as F
from nested_dict import nested_dict
from collections import OrderedDict
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


#这是一个残差模块，包含: 卷积-标准化-relu，卷积-标准化-relu+ x 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    #                BasicBlock            8   2
    def __init__(self, block,  J=2,N=32, k=1,n=2,num_classes=10,use_all=False):
        self.inplanes  = 16*k
        self.ichannels = 16*k#卷积输出特征数
        super(ResNet, self).__init__()
        self.nspace = int(N / (2 ** J))    #输入的一张图像的尺寸
        self.nfscat = int(1 + 8 * J + 8 * 8 * J * (J - 1) / 2)
        self.bn0 = nn.BatchNorm2d(self.nfscat*3,affine=False)#根据统计的mean和var来对数据进行标准化
        #定义卷积层
        self.conv1 = nn.Conv2d(self.nfscat*3,self.ichannels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        #标准化层
        self.bn1 = nn.BatchNorm2d(self.ichannels)
        #ReLU层
        self.relu = nn.ReLU(inplace=True)
        
            
        self.layer2 = self._make_layer(block, 32*k, n)#创建BasicBlock（残差模块），输出通道数32*k
        self.layer3 = self._make_layer(block, 64*k, n)#创建BasicBlock（残差模块），输出通道数64*k
        self.avgpool = nn.AvgPool2d(8)#池化窗口大小：8
        self.fc = nn.Linear(64*k, num_classes)#输出层

        #self.modules()采用深度优先遍历的方式，存储了net的所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    #创建BasicBlock（残差模块）
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #将通道数转为planes * block.expansion
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #输入的确实是（batch_size,3,81,8,8）
        #         batch_size    channels     N / (2 ** J) N / (2 ** J)  ;N = 32
        x = x.view(x.size(0), 3*self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)#BatchNorm
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
            
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)#(batch_size,...)

        x = self.fc(x)#输出层，num_classesg个元素
        return x



###########################################一层(不包含卷积)#####################################################
class JUST_FC(nn.Module):
    #                            8   2
    def __init__(self, J=2,N=32, k=1,n=2,num_classes=10,use_all=False):
        super(JUST_FC, self).__init__()
        self.nspace = 32    #输入的一张图像的尺寸
        self.nfscat = 73
        self.bn0 = nn.BatchNorm2d(self.nfscat*3,affine=False)#根据统计的mean和var来对数据进行标准化
               
        self.fc = nn.Linear(3*self.nfscat*32*32, num_classes)#输出层

        #self.modules()采用深度优先遍历的方式，存储了net的所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


    def forward(self, x):
        x = x.view(x.size(0), 3*self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)#BatchNorm

        x = x.view(x.size(0), -1)#(batch_size,...)

        x = self.fc(x)#输出层，num_classesg个元素
        return x  
        
                
def state_dict(module, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param
    for name, buf in module._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf
    for name, module in module._modules.items():
        if module is not None:
            state_dict(module, destination, prefix + name + '.')
    return destination


from torch.nn import Parameter
def params_stats(mod):
    params = OrderedDict()
    stats = OrderedDict()
    for k, v in state_dict(mod).items():
        if isinstance(v, Variable):
            params[k] = v
        else:
            stats[k] = v
    return params,stats

def resnet12_16_scat(N,J):
    return resnet12_scat(J,N,2,16)   
def resnet12_8_scat(N,J):
    return resnet12_scat(J,N,2,8)  #N = 32, J = 2
    
    
def resnet12_scat(J,N,n,k,use_all=False):
    """ Variant of WRN+Scattering used for 
    """
    #                         2 32 8 2
    model = ResNet(BasicBlock,J,N,k,n,use_all=use_all)
    
    model=model.cuda()
    params,stats=params_stats(model)

    return model, params, stats


#一层(只有FC)   
def just_fc(J, N, use_all=False):
    #
    model = JUST_FC(J,N,8,2, use_all=use_all)
    
    model=model.cuda()
    params,stats=params_stats(model)

    return model, params, stats
    

if __name__ == '__main__':
    resnet12_8_scat(32,2)
