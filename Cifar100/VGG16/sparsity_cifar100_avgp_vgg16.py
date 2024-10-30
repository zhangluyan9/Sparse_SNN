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

from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/experiment_1')
#1 3 7 15
q_level = 6
#load_name = "vgg16_T7_sparsity_L2_0dot01_step2_1e-9.pt"
#load_name = "baseline_T6/baseline_T6.pt"
#load_name = "vgg13_chaoyao_cifar100.pt"
load_name = 'baseline_T6_VGG13_cifar100.pt'
#load_name = "2bits_experiments/VGG16_chenyaofo_relu_10more_t3_q2_I0_30more_ours3e-8_01_1.pt"
#load_name = "2bits_experiments/VGG16_chenyaofo_relu_10more_t3_q2_I0_30more_ours3e-8_withNoise_newinput.pt"
save_name = "baseline_T6_VGG16_step100dot05_step2.pt"
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

class CalculateLoss(torch.nn.Module):
    def __init__(self, q_level,c_max):
        super(CalculateLoss, self).__init__()
        self.q_level = q_level
        self.max_value=c_max

    def forward(self, x):
        q_level = self.q_level
        c_max = self.max_value
        # Simplifying calculations by combining conditions and using in-place operations where possible
        Safe_zero_mask = (x <= 0)
        Safe_one_mask =  (x >= c_max + 0.5/q_level)

        x_scaled = x * q_level
        k = 2 * torch.round(x_scaled - 0.5 - 1e-5) + 1  # Finds the nearest odd integer to x_scaled
        seq_val = (k * 0.5) / q_level

        # Using torch.where to combine operations and reduce memory usage
        seq_val = torch.where((x >= 0) &(x <= 1/q_level), 0, seq_val)
        seq_val = torch.where(Safe_zero_mask, x, seq_val)
        seq_val = torch.where(Safe_one_mask, x, seq_val)
        x = torch.where((x >= 0) &(x <= 1/q_level), 0.5*x, x)
        # Loss calculation
        act_loss =  torch.sum(torch.abs(x - seq_val)) 

        #act_loss = (0.5 / q_level) ** 2 * torch.mean(torch.pow((torch.abs(x - seq_val) + 1e-10) / (0.5 / q_level), 0.5))

        return act_loss


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        #print(torch.randn(tensor.size()) * self.std + self.mean)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class PoolAndClamp(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(PoolAndClamp, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.clamp = Clamp_q_()  # Assuming Clamp_q_ is a module you've defined elsewhere

    def forward(self, x):
        x = self.pool(x)
        x = self.clamp(x)
        return x

class VGG_16(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=True):
        super(VGG_16, self).__init__()
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.cal = CalculateLoss(q_level,1)
        self.cq = Clamp_q_()
        self.classifier2 = nn.Linear(512 , 100)
    
    def forward(self, x):
        accumulated_y = 0
        layer_counter = 0
        for layer in self.features:
            #print(layer, str(layer) == 'Clamp_q_()')
            #print(layer,'AvgPool2d' in str(layer))
            x = layer(x)
            if 'AvgPool2d' in str(layer):
                x = self.cq(x)
            if str(layer) == 'Clamp_q_()':     
                y = torch.sum(x)
                accumulated_y += y#*1/(layer_counter+1)
                layer_counter+=1

        x = x.view(x.size(0), -1)
        x = self.classifier2(x)
        return x, accumulated_y

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=self.bias),nn.BatchNorm2d(out_channels),Clamp_q_()]
                in_channels = out_channels

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
          

def orthogonality_loss(model, beta):
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.size()) > 1:
            if len(param.size()) == 4:  # 卷积层权重，形状为 [out_channels, in_channels, kernel_height, kernel_width]

                W = param.view(param.size(0), -1)

            else:
                # 全连接层或其他二维权重
                W = param

            WT_W = torch.matmul(W.T, W)  # 计算 W^T * W
            I = torch.eye(WT_W.size(0), device=param.device)  # 创建单位矩阵并移到相同的设备
            #reg_loss += (WT_W - 0.00000000*I).pow(2).sum()  # 计算Frobenius范数的平方

            reg_loss += (WT_W - step1_factor * I).pow(2).sum()  # 计算Frobenius范数的平方

    return beta / 2.0 * reg_loss

def train(args, model, device, train_loader, optimizer, epoch,noise,use_function,clamp_max,scale_fa):
    noise = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        data.requires_grad = True
        output,sum_we = model(data)
        #print(sum_we)
        optimizer.zero_grad()
        #print("111111111111111111111111111111111111111111111111111111")
        #output,l_2 = model(data)
        if train_step[0]:
            loss  = F.cross_entropy(output, target) + sum_we*sc_f
        if train_step[1]:
            loss  = orthogonality_loss(model,2e-3)+F.cross_entropy(output, target) + l_2*step2_factor
        if train_step[2]:
            loss  = F.cross_entropy(output, target) #+ orthogonality_loss(model,2e-3)   


        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            #print(orthogonality_loss(model,2e-3),F.cross_entropy(output, target),sum_we*1e-7)

            if args.dry_run:
                break


def test(model, device, test_loader,noise,use_function,clamp_max):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i in range(1):
            correct_l = []
            for ti in range(1):
                correct = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    #onehot = torch.nn.functional.one_hot(target, 10)
                    output,sum_= model(data)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    #print(pred.eq(target.view_as(pred)).sum().item())
                    correct += pred.eq(target.view_as(pred)).sum().item()
                correct_l.append(correct)
            correct_l_n = np.array(correct_l)
            mean_val = np.mean(correct_l_n) / 1
            min_val = (np.min(correct_l_n) - np.mean(correct_l_n)) / 1
            max_val = (np.max(correct_l_n) - np.mean(correct_l_n)) / 1
            print(f"{mean_val:.2f}({min_val:.2f}, {max_val:.2f})",sum_)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct


def fuse_bn_sequential(block):
    if not isinstance(block, nn.Sequential):
        return block
    stack = []
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1], nn.Conv2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                if 'weight' in bn_st_dict:
                    gamma = bn_st_dict['weight']
                else:
                    gamma = torch.ones(mu.size(0)).float().to(mu.device)

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)

    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

    return model



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
    
    for k in range(5):
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
    test_loader = torch.utils.data.DataLoader(dataset2,batch_size=512) 

    model = VGG_16('VGG16', clamp_max=1,bias =True).to(device)
    #model_ = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    #torch.save(model_.state_dict(),load_name)
    model.load_state_dict(torch.load(load_name), strict=False)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay = weight_d_f)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.2)


    ACC = 0
    #test(model, device, test_loader, args.noise,args.use_function,args.clamp_max)
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    test(model, device, test_loader, args.noise,args.use_function,args.clamp_max)

    model = fuse_bn_recursively(model)
    torch.save(model.state_dict(),'merge_.pt')
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    test(model, device, test_loader, args.noise,args.use_function,args.clamp_max)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.noise,args.use_function,args.clamp_max,args.loss_scale)
        ACC_ = test(model, device, test_loader, args.noise,args.use_function,args.clamp_max)
        #epsilon = 8/255 
        #ACC_ = test_att(model, device, test_loader, epsilon)
        #print(epoch)
        if ACC_>ACC and epoch>70:
            ACC = ACC_
            torch.save(model.state_dict(),save_name)
        scheduler.step()

    #epsilon = 8/255
    #acc = test_att(model, device, test_loader, epsilon)




if __name__ == '__main__':
    main()





