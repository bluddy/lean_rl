import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.models as tmodels
import torch.nn.functional as F

feat_size = 7


def calc_features(img_stack):
    return img_stack

def make_linear(in_size, out_size, bn=False, drop=False, relu=True):
    l = []
    l.append(nn.Linear(in_size, out_size))
    if relu:
        l.append(nn.ReLU(inplace=True))
    if bn:
        l.append(nn.BatchNorm1d(out_size))
    if drop:
        l.append(nn.Dropout(p=0.1))
    return l

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def init_layers(layers):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer.weight.data.uniform_(*hidden_init(layer))
    #layers[-1].data.uniform_(-3e-3, 3e-3)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def make_conv(in_channels, out_channels, kernel_size, stride, padding, bn=False, drop=False):
    l = []
    l.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    l.append(nn.ReLU(inplace=True))
    l.append(nn.BatchNorm2d(out_channels))
    if drop:
        l.append(nn.Dropout2d(0.2))
    return l

class BaseImage(nn.Module):
    def __init__(self, img_stack, drop=False, img_dim=224, **kwargs):
        super(BaseImage, self).__init__()

        ## input size:[img_stack, 224, 224]
        print "BaseImage. drop:{}".format(drop)
        bn=True

        ll = []
        in_f = calc_features(img_stack)
        if img_dim == 224:
            d = 16
            ll.extend(make_conv(in_f, d,  3, 1, 1, bn=bn, drop=drop))
            ll.extend(make_conv(d,    d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 32
            ll.extend(make_conv(d,   d2,  3, 2, 1, bn=bn, drop=drop)) # 112
            ll.extend(make_conv(d2,  d2,  3, 1, 1, bn=bn, drop=drop))
            d = 64
            ll.extend(make_conv(d2,   d,  3, 2, 1, bn=bn, drop=drop)) # 56
            ll.extend(make_conv(d,    d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 128
            ll.extend(make_conv(d,   d2,  3, 2, 1, bn=bn, drop=drop)) # 28
            ll.extend(make_conv(d2,  d2,  3, 1, 1, bn=bn, drop=drop))
            d = 256
            ll.extend(make_conv(d2,   d,  3, 2, 1, bn=bn, drop=drop)) # 14
            ll.extend(make_conv(d,    d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 512
            ll.extend(make_conv(d,   d2,  3, 2, 1, bn=bn, drop=drop)) # 7
            d = 1024
            ll.extend(make_conv(d2,   d,  3, 2, 1, bn=bn, drop=drop)) # 4
            d2 = 2048
            ll.extend(make_conv(d,   d2,  3, 2, 1, bn=bn, drop=drop)) # 2

            self.latent_dim = 2 * 2 * 2048
        elif img_dim == 64:
            d = 16
            ll.extend(make_conv(in_f, d,  3, 1, 1, bn=bn, drop=drop))
            ll.extend(make_conv(d,    d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 32
            ll.extend(make_conv(d,   d2,  3, 2, 1, bn=bn, drop=drop)) # 64
            ll.extend(make_conv(d2,  d2,  3, 1, 1, bn=bn, drop=drop))
            d = 64
            ll.extend(make_conv(d2,   d,  3, 2, 1, bn=bn, drop=drop)) # 32
            ll.extend(make_conv(d,    d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 128
            ll.extend(make_conv(d,   d2,  3, 2, 1, bn=bn, drop=drop)) # 16
            ll.extend(make_conv(d2,  d2,  3, 1, 1, bn=bn, drop=drop))
            d = 256
            ll.extend(make_conv(d2,   d,  3, 2, 1, bn=bn, drop=drop)) # 8
            ll.extend(make_conv(d,    d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 512
            ll.extend(make_conv(d,   d2,  3, 2, 1, bn=bn, drop=drop)) # 4
            d = 1024
            ll.extend(make_conv(d2,   d,  3, 2, 1, bn=bn, drop=drop)) # 2

            self.latent_dim = 2 * 2 * 1024
        else:
            raise ValueError(str(img_dim) + " is not a valid img-dim")

        ll.extend([Flatten()])
        self.features = nn.Sequential(*ll)

class ActorImage(BaseImage):
    def __init__(self, action_dim, bn=False, **kwargs):
        super(ActorImage, self).__init__(bn=bn, **kwargs)

        ll = []
        ll.extend(make_linear(self.latent_dim, 400, bn=bn))
        ll.extend(make_linear(400, 100, bn=bn))
        self.linear = nn.Sequential(*ll)

        self.out_angular = nn.Linear(100, action_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        x = self.out_angular(x)
        #x = torch.clamp(x, min=-1., max=1.)
        x = torch.tanh(x)
        return x

class CriticImage(BaseImage):
    def __init__(self, action_dim, bn=False, **kwargs):
        super(CriticImage, self).__init__(bn=bn, **kwargs)

        ll = []
        ll.extend(make_linear(self.latent_dim + action_dim, 400, bn=bn))
        ll.extend(make_linear(400, 100, bn=bn))
        ll.extend(make_linear(100, 1, bn=False, relu=False))
        self.linear = nn.Sequential(*ll)

    def forward(self, x, u):
        x = self.features(x)
        x = torch.cat([x, u], 1)
        x = self.linear(x)
        return x

class QImage(BaseImage):
    def __init__(self, action_dim, bn=True, drop=False, **kwargs):
        super(QImage, self).__init__(bn=bn, drop=drop, **kwargs)

        ll = []
        #ll.extend(make_linear(self.latent_dim, action_dim, bn=bn, drop=drop))
        ll.extend(make_linear(self.latent_dim, action_dim, bn=False, drop=False, relu=False))
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x

class QImage2Outs(BaseImage):
    ''' QImage with two outputs coming out of the featres
        Use state shape of QState
    '''
    def __init__(self, action_dim, aux_size, drop=False, reduced_dim=10, **kwargs):
        super(QImage2Outs, self).__init__(drop=drop, **kwargs)

        print "QImage2Outs: reduced_dim={}, drop={}".format(reduced_dim, drop)

        bn=True
        d = reduced_dim

        # Map features to small state space
        ll = []
        ll.extend(make_linear(self.latent_dim, d, bn=bn, drop=drop))
        ll.extend(make_linear(d, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 50, bn=bn, drop=drop))
        self.linear = nn.Sequential(*ll)

        # RL part
        ll = []
        ll.extend(make_linear(50, action_dim, drop=False, bn=False, relu=False))
        self.linear1 = nn.Sequential(*ll)

        # Aux part
        ll = []
        ll.extend(make_linear(50, aux_size, bn=False, drop=False, relu=False))
        self.linear2 = nn.Sequential(*ll)

    def freeze_some(self, frozen):
        raise ValueError("Freezing not supported in QImage2Outs")

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        y = self.linear1(x)
        z = self.linear2(x)
        return y,z


class QImage2OutsFreeze(BaseImage):
    ''' QImage with two outputs coming out of the features
        Allow freezing of CNN part
        Use state shape of QState
    '''
    def __init__(self, action_dim, aux_size, drop=False, reduced_dim=10, **kwargs):
        super(QImage2Outs, self).__init__(drop=drop, **kwargs)

        print "QImage2Outs: reduced_dim={}, drop={}".format(reduced_dim, drop)

        bn=True
        d = reduced_dim

        # Map features to small state space
        ll = []
        ll.extend(make_linear(self.latent_dim, d, bn=bn, drop=drop))
        self.linear = nn.Sequential(*ll)

        # RL part
        ll = []
        ll.extend(make_linear(d, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 50, bn=bn, drop=drop))
        ll.extend(make_linear(50, action_dim, drop=False, bn=False, relu=False))
        self.linear1 = nn.Sequential(*ll)

        # Aux part
        ll = []
        ll.extend(make_linear(d, aux_size, bn=False, drop=False, relu=False))
        self.linear2 = nn.Sequential(*ll)

    def freeze_some(self, frozen):
        for p in self.linear.parameters():
            p.requires_grad = not frozen
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        y = self.linear1(x)
        z = self.linear2(x)
        return y,z

class QImageDenseNet(nn.Module):
    def __init__(self, action_dim, img_stack, pretrained=False):
        super(QImageDenseNet, self).__init__()

        print "QImageDenseNet. action_dim:{}, img_stack:{}".format(action_dim, img_stack)

        model = tmodels.densenet121(pretrained=False)
        layers = list(model.features.children())
        if img_stack != 3:
            layers[0] = nn.Conv2d(img_stack, 64, 7, stride=2, padding=3, bias=False)

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(1024, action_dim)

    def forward(self, x):
        #import pdb
        #pdb.set_trace()

        x = F.relu(self.features(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class QMixedDenseNet(QImageDenseNet):
    def __init__(self, action_dim, state_dim, img_stack):
        self.latent_dim = 100
        super(QMixedDenseNet, self).__init__(action_dim=self.latent_dim, img_stack=img_stack)

        print "QMixedDenseNet. action_dim:{}, img_stack:{}, state_dim:{}".format(action_dim, img_stack, state_dim)

        bn = True
        drop = False

        ll = []
        ll.extend(make_linear(self.latent_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 100, bn=bn, drop=drop))
        self.linear1 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(state_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 100, bn=bn, drop=drop))
        self.linear2 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(200, action_dim, bn=False, drop=False, relu=False))
        self.linear3 = nn.Sequential(*ll)

    def forward(self, x):
        img, state = x
        x = self.model(img)
        x = self.linear1(x)
        y = self.linear2(state)
        z = self.linear3(torch.cat((x, y), dim=-1))
        return z

class ActorState(nn.Module):
    def __init__(self, state_dim, action_dim, bn=False):
        super(ActorState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend(make_linear(300, 100, bn=bn))
        ll.extend(make_linear(100, action_dim, bn=False, drop=False, relu=False))

        # init
        init_layers(ll)

        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.linear(x)
        #x = torch.clamp(x, min=-1., max=1.)
        x = torch.tanh(x)
        return x


class CriticState(nn.Module):
    def __init__(self, state_dim, action_dim, bn=False):
        super(CriticState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim + action_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend(make_linear(300, 100, bn=bn))
        ll.extend(make_linear(100, 1, bn=False, drop=False, relu=False))

        init_layers(ll)

        self.linear = nn.Sequential(*ll)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = self.linear(x)
        return x

class QState(nn.Module):
    def __init__(self, state_dim, action_dim, bn=True, drop=False, **kwargs):
        super(QState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 50, bn=bn, drop=drop))
        ll.extend(make_linear(50, action_dim, drop=False, bn=False, relu=False))

        #init_layers(ll)

        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.linear(x)
        return x

class QMixed(BaseImage):
    def __init__(self, state_dim, action_dim,
            bn=True, drop=False, **kwargs):
        super(QMixed, self).__init__(bn=bn, drop=drop, **kwargs)

        # Map features
        ll = []
        ll.extend(make_linear(self.latent_dim, 50, bn=bn, drop=drop))
        ll.extend(make_linear(50, 20, bn=bn, drop=drop))
        self.linear1 = nn.Sequential(*ll)

        # Map state
        ll = []
        ll.extend(make_linear(state_dim, 50, bn=bn, drop=drop))
        ll.extend(make_linear(50, 20, bn=bn, drop=drop))
        self.linear2 = nn.Sequential(*ll)

        # Combine
        ll = []
        ll.extend(make_linear(40, action_dim, bn=False, drop=False, relu=False))
        self.linear3 = nn.Sequential(*ll)

    def forward(self, x):
        img, state = x
        x = self.features(img)
        x = self.linear1(x)
        y = self.linear2(state)
        x = self.linear3(torch.cat((x, y), dim=-1))
        return x

class QMixed2Outs(BaseImage):
    ''' QMixed with two outputs coming out of the features
        Use state shape of QState
    '''
    def __init__(self, state_dim, action_dim, aux_size,
            drop=False, reduced_dim=10, **kwargs):
        super(QMixed2Outs, self).__init__(drop=drop, **kwargs)

        print "QMixed2Outs: reduced_dim={}, drop={}".format(reduced_dim, drop)

        bn=True
        d = reduced_dim

        # Map features to small state space
        ll = []
        ll.extend(make_linear(self.latent_dim, d, bn=bn, drop=drop))
        self.linear = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(d + state_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 50, bn=bn, drop=drop))
        self.linear1 = nn.Sequential(*ll)

        # RL part
        ll = []
        ll.extend(make_linear(50, action_dim, drop=False, bn=False, relu=False))
        self.linear2 = nn.Sequential(*ll)

        # Aux part
        ll = []
        ll.extend(make_linear(50, aux_size, bn=False, drop=False, relu=False))
        self.linear3 = nn.Sequential(*ll)

    def freeze_some(self, frozen):
        raise ValueError("Freezing not supported in QMixed2Outs")

    def forward(self, x):
        img, state = x
        # Features to state
        x = self.features(img)
        x = self.linear(x)
        x = self.linear1(torch.cat((x, state), dim=-1)) # RL output
        y = self.linear2(x) # RL output
        z = self.linear3(x) # Aux output
        return y,z


class QMixed2OutsFreeze(BaseImage):
    ''' QMixed with two outputs coming out of the features
        Needs freezing
        Use state shape of QState
    '''
    def __init__(self, state_dim, action_dim, aux_size,
            drop=False, reduced_dim=10, **kwargs):
        super(QMixed2OutsFreeze, self).__init__(drop=drop, **kwargs)

        print "QMixed2OutsFreeze: reduced_dim={}, drop={}".format(reduced_dim, drop)

        bn=True
        d = reduced_dim

        # Map features to small state space
        ll = []
        ll.extend(make_linear(self.latent_dim, d, bn=bn, drop=drop))
        self.linear = nn.Sequential(*ll)

        # RL part
        ll = []
        ll.extend(make_linear(d + state_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 50, bn=bn, drop=drop))
        ll.extend(make_linear(50, action_dim, drop=False, bn=False, relu=False))
        self.linear1 = nn.Sequential(*ll)

        # Aux part
        ll = []
        ll.extend(make_linear(d, aux_size, bn=False, drop=False, relu=False))
        self.linear2 = nn.Sequential(*ll)

    def freeze_some(self, frozen):
        for p in self.linear.parameters():
            p.requires_grad = not frozen
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def forward(self, x):
        img, state = x
        # Features to state
        x = self.features(img)
        x = self.linear(x)
        y = self.linear1(torch.cat((x, state), dim=-1)) # RL output
        z = self.linear2(x) # Aux output
        return y,z


class QMixed2(nn.Module):
    def __init__(self, img_stack, bn=True, drop=False, img_dim=224, deep=False):
        super(QMixed2, self).__init__()

        ## input size:[img_stack, 224, 224]
        print "QMixed2. drop:{}, deep:{}, bn:{}".format(drop, deep, bn)

        ll = []
        in_f = calc_features(img_stack)
        if img_dim == 224:
            d = 4; l = img_dim
            ll.extend(make_conv(in_f, d,  1, 1, 1, bn=bn, drop=drop)) # flatten colors, 224
            d2 = 8; l = l / 2
            ll.extend(make_conv(d, d2, 3, 2, 1, bn=bn, drop=drop)) # 112
            if deep:
                ll.extend(make_conv(d2, d2, 3, 1, 1, bn=bn, drop=drop))
            d = 16; l = l / 2
            ll.extend(make_conv(d2, d,  3, 2, 1, bn=bn, drop=drop)) # 56
            if deep:
                ll.extend(make_conv(d, d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 32; l = l / 2
            ll.extend(make_conv(d,  d2,  3, 2, 1, bn=bn, drop=drop)) # 28
            if deep:
                ll.extend(make_conv(d2, d2,  3, 1, 1, bn=bn, drop=drop))
            d = 64; l = l / 2
            ll.extend(make_conv(d2,  d,  3, 2, 1, bn=bn, drop=drop)) # 14
            if deep:
                ll.extend(make_conv(d, d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 128; l = l / 2
            ll.extend(make_conv(d, d2,  3, 2, 1, bn=bn, drop=drop)) # 7
            if deep:
                ll.extend(make_conv(d2, d2,  3, 1, 1, bn=bn, drop=drop))
            d = d2
            self.latent_dim = l * l * d
        else:
            raise ValueError(str(img_dim) + " is not a valid img_dim")

        ll.extend([Flatten()])
        self.features = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(self.latent_dim, 400, bn=bn, drop=drop))
        ll.extend(make_linear(400, 100, bn=bn, drop=drop))
        self.linear1 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(state_dim, 400, bn=bn, drop=drop))
        ll.extend(make_linear(400, 100, bn=bn, drop=drop))
        self.linear2 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(200, action_dim, bn=False, drop=False, relu=False))
        self.linear3 = nn.Sequential(*ll)

    def forward(self, x):
        img, state = x
        x = self.features(img)
        x = self.linear1(x)
        y = self.linear2(state)
        x = self.linear3(torch.cat((x, y), dim=-1))
        return x

class QImageSoftMax(BaseImage):
    ''' Image network with softmax '''
    def __init__(self, action_dim, bn=True, drop=False, **kwargs):
        super(QImageSoftMax, self).__init__(bn=bn, drop=drop, **kwargs)

        ll = []
        ll.extend(make_linear(self.latent_dim, 400, bn=bn, drop=drop))
        ll.extend(make_linear(400, action_dim, bn=False, drop=False, relu=False))
        ll.extend([nn.LogSoftmax(dim=-1)])
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x
