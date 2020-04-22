import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.models as tmodels

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
        l.append(nn.Dropout(p=0.2))
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
    if bn:
        l.append(nn.BatchNorm2d(out_channels))
    if drop:
        l.append(nn.Dropout2d(0.2))
    return l

class BaseImage(nn.Module):
    def __init__(self, img_stack, bn=True, drop=False, img_dim=224, deep=False):
        super(BaseImage, self).__init__()

        ## input size:[img_stack, 224, 224]
        print "BaseImage. drop:{}, deep:{}, bn:{}".format(drop, deep, bn)

        ll = []
        in_f = calc_features(img_stack)
        if img_dim == 224:
            ll.extend(make_conv(in_f, 16,  3, 1, 1, bn=bn, drop=drop))
            if deep:
                ll.extend(make_conv(16, 16,  3, 1, 1, bn=bn, drop=drop))
            ll.extend(make_conv(16,   32,  3, 2, 1, bn=bn, drop=drop)) # 112
            if deep:
                ll.extend(make_conv(32, 32,  3, 1, 1, bn=bn, drop=drop))
            ll.extend(make_conv(32,   64,  3, 2, 1, bn=bn, drop=drop)) # 56
            if deep:
                ll.extend(make_conv(64, 64,  3, 1, 1, bn=bn, drop=drop))
            ll.extend(make_conv(64,  128,  3, 2, 1, bn=bn, drop=drop)) # 28
            if deep:
                ll.extend(make_conv(128, 128,  3, 1, 1, bn=bn, drop=drop))
            ll.extend(make_conv(128, 256,  3, 2, 1, bn=bn, drop=drop)) # 14
            if deep:
                ll.extend(make_conv(256, 256,  3, 1, 1, bn=bn, drop=drop))
            ll.extend(make_conv(256, 512,  3, 2, 1, bn=bn, drop=drop)) # 7
            self.latent_dim = 7 * 7 * 512
        elif img_dim == 64:
            ll.extend(make_conv(in_f, 16,  3, 1, 1, bn=bn, drop=drop)),
            ll.extend(make_conv(16,   32,  3, 2, 1, bn=bn, drop=drop)), # 32
            ll.extend(make_conv(32,   64,  3, 2, 1, bn=bn, drop=drop)), # 16
            ll.extend(make_conv(64,  128,  3, 2, 1, bn=bn, drop=drop)), # 8
            ll.extend(make_conv(128, 256,  3, 2, 1, bn=bn, drop=drop)), # 4
            self.latent_dim = 4 * 4 * 256
        else:
            raise ValueError(str(img_dim) + " is not a valid img-dim")

        ll.extend([Flatten()])
        self.features = nn.Sequential(*ll)

class ImageToPos(BaseImage):
    ''' Class converting the image to a position of the needle.
        We train on this to accelerate RL training off images
    '''
    def __init__(self, img_stack, out_size=3, bn=False, img_dim=224):
        super(ImageToPos, self).__init__(img_stack, bn, img_dim=img_dim)

        ll = []
        ll.extend(make_linear(self.latent_dim, 400, bn=bn))
        ll.extend(make_linear(400, out_size, bn=False, relu=False)) # x, y, w
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x

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
        ll.extend(make_linear(self.latent_dim, 400, bn=bn, drop=drop))
        ll.extend(make_linear(400, action_dim, bn=False, drop=False, relu=False))
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x

class QImageDenseNet(nn.Module):
    def __init__(self, action_dim, img_stack, img_dim=224):
        assert (img_stack==1)
        super(QImageDenseNet, self).__init__()
        self.model = tmodels.densenet121(pretrained=True)
        c = self.model.classifier
        in_features = c.in_features
        self.model.classifier = nn.Linear(c.in_features, action_dim)

    def forward(self, x):
        x = self.model(x)
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
