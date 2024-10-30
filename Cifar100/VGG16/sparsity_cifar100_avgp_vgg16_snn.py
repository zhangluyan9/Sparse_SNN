from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import MultiStepLR
import catSNN
import catCuda
from catSNN import spikeLayer, load_model, max_weight, normalize_weight, SpikeDataset, fuse_module
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/experiment_1')
#1 3 7 15
q_level = 6
#load_name = "vgg16_T7_sparsity_L2_0dot01_step2_1e-9.pt"
load_name = 'merge_.pt'
#load_name = "2bits_experiments/VGG16_chenyaofo_relu_10more_t3_q2_I0_30more_ours3e-8_01_1.pt"
#load_name = "2bits_experiments/VGG16_chenyaofo_relu_10more_t3_q2_I0_30more_ours3e-8_withNoise_newinput.pt"
#save_name = "vgg16_baseline_T7_snntype_test.pt"
weight_d_f = 0
sc_f = 1e-6

train_step = [True,False,False]
#train_step = [False,True,False]
#train_step = [False,False,True]

step1_factor = 0
step1_factor_ = 1e-1
#print(step1_factor*10)
#4:5e-8
#5:3e-8
#6:7e-8
step2_factor = 5e-9
cfg = {
    'o' : [128,128,'M',256,256,'M',512,512,'M',(1024,0),'M'],
    'VGG13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}




class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=1):
        ctx.constant = constant
        new_x = torch.div(torch.floor(torch.mul(tensor, constant)), constant)
        return new_x

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return 100*F.hardtanh(grad_output/100), None 

Quantization_ = Quantization.apply

class Clamp_q_(nn.Module):
    def __init__(self, min=0.0, max=1,q_level = q_level):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max
        self.q_level = q_level

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = Quantization_(x, self.q_level)
        return x

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    #1-0.00001
    out = catCuda.getSpikes(out, 1-0.0001)
    return out

class NewSpike(nn.Module):
    def __init__(self, T = q_level):
        super(NewSpike, self).__init__()
        self.T = T

    def forward(self, x):

        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        return x


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        #print(torch.randn(tensor.size()) * self.std + self.mean)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class VGG_16(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=True):
        super(VGG_16, self).__init__()
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.cq = Clamp_q_()
        self.classifier2 = nn.Linear(512 , 100,bias = True)
        
    
    def forward(self, x):
        accumulated_y = 0
        layer_counter = 0
        for layer in self.features:
            x = layer(x)
            if 'AvgPool2d' in str(layer):
                x = self.cq(x)
        x = x.view(x.size(0), -1)
        x = self.classifier2(x)
        return x, accumulated_y,1

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),Clamp_q_()]
                in_channels = out_channels

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class CatVGG_o(nn.Module):
    def __init__(self, vgg_name, T, is_noise=False, bias=True):
        super(CatVGG_o, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.is_noise = is_noise
        self.bias = bias

        self.features = self._make_layers(cfg[vgg_name], is_noise)
        self.classifier2 =self.snn.dense((1, 1, 512), 100,bias = True)

    def forward(self, x):
        accumulated_y = 0
        layer_shape = 0
        for layer in self.features:
            #print(layer)
            x = layer(x)
            if str(layer) == 'Clamp_q_()':
                x = create_spike_input_cuda(x, q_level)
                #accumulated_y+=torch.sum(x)
            if  '_poolLayer' in str(layer):
                x = (torch.sum(x, dim=4))/self.T
                x = create_spike_input_cuda(x, q_level)
                accumulated_y+=torch.sum(x)
                layer_shape+=x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
                #accumulated_y+=torch.sum(x)
            if  'NewSpike()' in str(layer):
                accumulated_y+=torch.sum(x)
                layer_shape+=x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
                #print(x.shape)

        #out = self.features(x)
        out = self.classifier2(x)
        out = self.snn.sum_spikes(out) / self.T
        return out,accumulated_y,layer_shape

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        in_channels = 3
        first_conv = True  # 标记是否是第一个卷积层

        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(2)]
            else:
                if is_noise:
                    layers += [self.snn.mcConv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                               self.snn.spikeLayer(1.0),nn.Dropout2d(0)]
                    in_channels = x
                else:
                    padding = x[1] if isinstance(x, tuple) else 1
                    out_channels = x[0] if isinstance(x, tuple) else x
                    #IF
                    if first_conv:
                        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),
                                Clamp_q_()]
                        first_conv = False

                    else:
                        layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                                NewSpike(q_level)]
                    in_channels = out_channels
        return nn.Sequential(*layers)       




def test(model, device, test_loader,noise,use_function,clamp_max):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        sum_a = 0
        shape = 0
        for i in range(1):
            correct_l = []
            for ti in range(1):
                correct = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    #onehot = torch.nn.functional.one_hot(target, 10)
                    output,sum_,shape_= model(data)
                    sum_a+=sum_
                    shape+=shape_
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    #print(pred.eq(target.view_as(pred)).sum().item())
                    correct += pred.eq(target.view_as(pred)).sum().item()
                correct_l.append(correct)
            correct_l_n = np.array(correct_l)
            mean_val = np.mean(correct_l_n) / 1
            min_val = (np.min(correct_l_n) - np.mean(correct_l_n)) / 1
            max_val = (np.max(correct_l_n) - np.mean(correct_l_n)) / 1
            print(f"{mean_val:.2f}({min_val:.2f}, {max_val:.2f})",sum_a/(shape*q_level)*100,sum_a/shape)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct


def transfer_model(src, dst, quantize_bit=32):
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()
    reshape_dict = {}
    for (k, v) in src_dict.items():
        if k in dst_dict.keys():
            reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape))
    dst.load_state_dict(reshape_dict, strict=False)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--loss_scale', type=float, default=0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--noise', type=float, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--use_function', type=str, default='relu', metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--clamp_max', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--T', type=int, default=5, metavar='N',
                        help='SNN time window')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 32,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding = 4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    transform_clean=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset1 = datasets.CIFAR100('../data', train=True, download=True,
                       transform=transform_train)
    
    for k in range(0):
        for i in range(1):
            transform_train_1 = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomCrop(32, padding = 6),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AddGaussianNoise(std=0.01),
                #AddQuantization()
            ])

        dataset1 = dataset1+ datasets.CIFAR100('../data', train=True, download=True,
                       transform=transform_train_1)
    
    dataset2 = datasets.CIFAR100('../data', train=False,
                       transform=transform)
    #print(type(dataset1[0][0]))
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2,batch_size=1000) 

    model = VGG_16('VGG16', clamp_max=1,bias =True).to(device)
    model.load_state_dict(torch.load(load_name), strict=False)
    model_snn = CatVGG_o('VGG16', T= q_level,bias =True).to(device)
    transfer_model(model,model_snn)
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #for param_tensor in model_snn.state_dict():
    #    print(param_tensor, "\t", model_snn.state_dict()[param_tensor].size())

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay = weight_d_f)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.2)


    ACC = 0
    #test(model, device, test_loader, args.noise,args.use_function,args.clamp_max)
    test(model, device, test_loader, args.noise,args.use_function,args.clamp_max)
    test(model_snn, device, test_loader, args.noise,args.use_function,args.clamp_max)
    #model = fuse_bn_recursively(model)
    #torch.save(model.state_dict(),'merge.pt')

    

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.noise,args.use_function,args.clamp_max,args.loss_scale)
        ACC_ = test(model, device, test_loader, args.noise,args.use_function,args.clamp_max)
        #epsilon = 8/255 
        #ACC_ = test_att(model, device, test_loader, epsilon)
        if ACC_>ACC or ACC_ == ACC:
            ACC = ACC_
            torch.save(model.state_dict(),save_name)
        scheduler.step()

    #epsilon = 8/255
    #acc = test_att(model, device, test_loader, epsilon)




if __name__ == '__main__':
    main()





