import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler


####################################################################
# ------------------------- Discriminators --------------------------
####################################################################
class Dis(nn.Module):
    def __init__(self, input_dim, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        n_layer = 5
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs


class MultiScaleDis(nn.Module):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
        super(MultiScaleDis, self).__init__()
        ch = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
        tch = ch
        for _ in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for Dis in self.Diss:
            outs.append(Dis(x))
            x = self.downsample(x)
        return outs


class Dis_content(nn.Module):
    def __init__(self, ndf=256):
        super(Dis_content, self).__init__()
        model = []
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(ndf, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs


####################################################################
# ---------------------------- Encoders -----------------------------
####################################################################
class E_content(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(E_content, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=ngf, kernel_size=7, stride=1, padding=3)
        self.Instance1 = nn.InstanceNorm2d(ngf)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=1)
        self.Instance2 = nn.InstanceNorm2d(ngf * 2)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=1)
        self.Instance3 = nn.InstanceNorm2d(ngf * 4)
        self.relu3 = nn.LeakyReLU(0.2, True)
        self.residual_layer = self.make_layer(Residule_Block, 4, ngf * 4)

    def make_layer(self, block, num_residule_block, num_channel):
        layers = []
        for _ in range(num_residule_block):
            layers.append(block(num_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x (3, 256, 256)
        out = self.conv1(x)
        out = self.Instance1(out)
        out = self.relu1(out)
        # x (64, 256, 256)
        out = self.conv2(out)
        out = self.Instance2(out)
        out = self.relu2(out)
        # x (64*2, 128, 128)
        out = self.conv3(out)
        out = self.Instance3(out)
        out = self.relu3(out)
        # x (64*4, 64, 64)
        out = self.residual_layer(out)
        # x (64*4, 64, 64)
        return out


class E_attr(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(E_attr, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=ngf, kernel_size=7, stride=1, padding=3)
        self.Instance1 = nn.InstanceNorm2d(ngf)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=1)
        self.Instance2 = nn.InstanceNorm2d(ngf * 2)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=1)
        self.Instance3 = nn.InstanceNorm2d(ngf * 4)
        self.relu3 = nn.LeakyReLU(0.2, True)
        self.residual_layer = self.make_layer(Residule_Block, 4, ngf * 4)

    def make_layer(self, block, num_residule_block, num_channel):
        layers = []
        for _ in range(num_residule_block):
            layers.append(block(num_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x (3, 256, 256)
        out = self.conv1(x)
        out = self.Instance1(out)
        out = self.relu1(out)
        # x (64, 256, 256)
        out = self.conv2(out)
        out = self.Instance2(out)
        out = self.relu2(out)
        # x (64*2, 128, 128)
        out = self.conv3(out)
        out = self.Instance3(out)
        out = self.relu3(out)
        # x (64*4, 64, 64)
        out = self.residual_layer(out)
        # x (64*4, 64, 64)
        return out


####################################################################
# --------------------------- Generators ----------------------------
####################################################################
class G(nn.Module):
    def __init__(self, output_dim, num_residule_block=5, ngf=64):
        super(G, self).__init__()

        self.residual_layer = self.make_layer(Residule_Block, num_residule_block, ngf * 8)
        self.up1 = functools.partial(nn.functional.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(ngf * 8, ngf * 2, kernel_size=3, stride=1, padding=1)
        self.Instance1 = nn.InstanceNorm2d(ngf * 2)
        self.relu1 = nn.ReLU(True)

        self.up2 = functools.partial(nn.functional.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(ngf * 2, ngf * 1, kernel_size=3, stride=1, padding=1)
        self.Instance2 = nn.InstanceNorm2d(ngf * 1)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(ngf * 1, output_dim, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

    def make_layer(self, block, num_residule_block, num_channel):
        layers = []
        for _ in range(num_residule_block):
            layers.append(block(num_channel))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        # x (ngf*8, 64, 64)
        out = self.residual_layer(out)
        # x (ngf*8, 64, 64)
        out = self.up1(out)
        out = self.conv1(out)
        out = self.Instance1(out)
        out = self.relu1(out)
        # x (ngf*2, 128, 128)
        out = self.up2(out)
        out = self.conv2(out)
        out = self.Instance2(out)
        out = self.relu2(out)
        # x (ngf*1, 256, 256)
        out = self.conv3(out)
        out = self.tanh(out)
        # x (3, 256, 256)
        return out


####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    elif opts.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler


def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fainplanes')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, gpu, init_type='normal', gain=0.02):
    assert (torch.cuda.is_available())
    net.to(gpu)
    init_weights(net, init_type, gain)
    return net


####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################
# conv + (spectral) + (instance) + leakyrelu
class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(
                nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Residule_Block(nn.Module):
    def __init__(self, nc):
        super(Residule_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance1 = nn.InstanceNorm2d(nc)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance2 = nn.InstanceNorm2d(nc)

    def forward(self, x):
        out = self.conv1(x)
        out = self.instance1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.instance2(out)
        return out + x
