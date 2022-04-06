import torch
import timm
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
from itertools import repeat
import collections.abc

def Tensor2Image(img):
    """
    input (FloatTensor)
    output (PIL.Image)
    """
    img = img.cpu()
    img = img * 0.5 + 0.5
    img = transforms.ToPILImage()(img)
    return img

def one_hot(label, depth):
    """
    Return the one_hot vector of the label given the depth.
    Args:
        label (LongTensor): shape(batchsize)
        depth (int): the sum of the labels

    output: (FloatTensor): shape(batchsize x depth) the label indicates the index in the output

    >>> label = torch.LongTensor([0, 0, 1])
    >>> one_hot(label, 2)
    <BLANKLINE>
     1  0
     1  0
     0  1
    [torch.FloatTensor of size 3x2]
    <BLANKLINE>
    """
    out_tensor = torch.zeros(len(label), depth)
    for i, index in enumerate(label):
        out_tensor[i][index] = 1
    return out_tensor

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.uniform_(m.weight.data, 0.02, 1)
        init.constant_(m.bias.data, 0.0)

      

class conv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 3, 96, 96))

    >>> net = conv_unit(3, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 96, 96])

    >>> net = conv_unit(3, 16, pooling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 48, 48])
    """

    def __init__(self, in_channels, out_channels, pooling=False):
        super(conv_unit, self).__init__()

        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Fconv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 64, 48, 48))

    >>> net = Fconv_unit(64, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 48, 48])

    >>> net = Fconv_unit(64, 16, unsampling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 96, 96])
    """

    def __init__(self, in_channels, out_channels, unsampling=False):
        super(Fconv_unit, self).__init__()

        if unsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Decoder(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_z (int): The dimensions of the noise

    >>> Dec = Decoder()
    >>> input = Variable(torch.randn(4, 372))
    >>> output = Dec(input)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self, N_p=2, N_z=50):
        super(Decoder, self).__init__()
        Fconv_layers = [
            Fconv_unit(320, 160),                   #Bx160x7x7
            Fconv_unit(160, 256),                   #Bx256x7x7
            Fconv_unit(256, 256, unsampling=True),  #Bx256x14x14
            Fconv_unit(256, 128),                   #Bx128x14x14
            Fconv_unit(128, 192),                   #Bx192x14x14
            Fconv_unit(192, 192, unsampling=True),  #Bx192x28x28
            Fconv_unit(192, 96),                    #Bx96x28x28
            Fconv_unit(96, 128),                    #Bx128x28x28
            Fconv_unit(128, 128, unsampling=True),  #Bx128x56x56
            Fconv_unit(128, 64),                    #Bx64x56x56
            Fconv_unit(64, 64),                     #Bx64x56x56
            Fconv_unit(64, 64, unsampling=True),    #Bx64x112x112
            Fconv_unit(64, 32),                     #Bx32x112x112
            Fconv_unit(32, 32),                     #Bx32x112x112
            Fconv_unit(32, 32, unsampling=True),    #Bx32x224x224
            Fconv_unit(32,3)                        #Bx3x224x224
        ]

        self.Fconv_layers = nn.Sequential(*Fconv_layers)
        self.mlp_layer = nn.Linear(768+N_p+N_z, 320*7*7)

    def forward(self, input):
        x = self.mlp_layer(input)
        x = x.view(-1, 320, 7, 7)
        x = self.Fconv_layers(x)
        return x



class Encoder(nn.Module):
    """
    The single version of the Encoder.

    >>> Enc = Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self):
        super().__init__()
        self.vit_feature_extract = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

    def forward(self, x):
        x = self.vit_feature_extract(x)
        return x


class Generator(nn.Module):
    """
    >>> G = Generator()

    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> pose = Variable(torch.randn(4, 2))
    >>> noise = Variable(torch.randn(4, 50))

    >>> output = G(input, pose, noise)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self, N_p=2, N_z=50, single=True):
        super(Generator, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder(N_p, N_z)

    def forward(self, input, pose, noise):
        x = self.enc(input)
        x = torch.cat((x, pose, noise), 1)
        x = self.dec(x)
        return x

class Discriminator(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_d (int): The sum of the identities

    >>> D = Discriminator()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = D(input)
    >>> output.size()
    torch.Size([4, 503])
    """
    def __init__(self, N_p=2, N_d=500):
        super(Discriminator, self).__init__()
        #Because Discriminator uses same architecture as that of Encoder
        self.enc = Encoder() 
        self.fc = nn.Linear(768, N_d+N_p+1)

    def forward(self,input):
        x = self.enc(input)
        x = x.view(-1, 768)
        x = self.fc(x)
        return x
