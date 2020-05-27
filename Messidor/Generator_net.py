import torch
import torch.nn as nn
import torch.nn.functional as F


class UGenerator_Net(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(UGenerator_Net, self).__init__()
        self.name = 'unet'
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 3, 2, 1)

        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 3, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 5, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 5, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 2, output_nc, 5, 2, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(ngf)
            self.norm2 = nn.BatchNorm2d(ngf * 2)
            self.norm4 = nn.BatchNorm2d(ngf * 4)
            self.norm8 = nn.BatchNorm2d(ngf * 8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(ngf)
            self.norm2 = nn.InstanceNorm2d(ngf * 2)
            self.norm4 = nn.InstanceNorm2d(ngf * 4)
            self.norm8 = nn.InstanceNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 299 x 299
        e1 = self.conv1(input)
        #print(e1.shape)
        # state size is (ngf) x 149 x 149
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        #print(e2.shape)
        # state size is (ngf) x 74 x 74
        e3 = self.norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 2) x 37 x 37

        #print(e3.shape)
        e4 = self.norm8(self.conv4(self.leaky_relu(e3)))
        #print(e4.shape)
        # state size is (ngf x 4) x 18 x 18

        e5 = self.norm8(self.conv5(self.leaky_relu(e4)))
        #: state size is (ngf x 8) x 9 x 9
        #print(e5.shape)
        e6 = self.conv6(self.leaky_relu(e5))
        # state size is (ngf x 8) x 5 x 5
        #print(e6.shape)
        #e7 = self.norm8(self.conv7(self.leaky_relu(e6)))
        #print(e7.shape)
        # state size is (ngf x 8) x 1 x 1

        # No batch norm on output of Encoder
        #e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 2 x 4
        #d1_ = self.dropout(self.norm8(self.dconv1(self.act(e8))))
        # state size is (ngf x 8) x 4 x 8
        #d1 = torch.cat((d1_, e7), 1)
        #d2_ = self.dropout(self.norm8(self.dconv2(self.act(d1))))
        #d2 = torch.cat((d2_, e6), 1)
        # state size is (ngf x 8) x 4 x 4
        d1_ = self.dropout(self.norm8(self.dconv1(self.act(e6))))
        #d1_ = self.norm8(self.dconv1(self.act(e6)))
        #print(d1_.shape)
        #print(e5.shape)
        # print(d1_.shape) : state size is (ngf x 8) x 9 x 9
        d1 = torch.cat((d1_, e5), 1)
        d2_ = self.norm8(self.dconv2(self.act(d1)))
        #print(d2_.shape) : state size is (ngf x 8) x 18 x 18
        d2 = torch.cat((d2_, e4), 1)
        #print(d2.shape)
        d3_ = self.norm4(self.dconv3(self.act(d2)))
        #print(d3_.shape)
        # state size is (ngf x 4) x 28 x 28
        # print(d3_.shape)
        # print(e3.shape)
        d3 = torch.cat((d3_, e3), 1)

        d4_ = self.norm2(self.dconv4(self.act(d3)))
        # state size is (ngf x 2) x 56 x 56
        d4 = torch.cat((d4_, e2), 1)
        d5_ = self.norm(self.dconv5(self.act(d4)))
        # state size is (ngf) x 112 x 112
        d5 = torch.cat((d5_, e1), 1)
        d6 = self.dconv6(self.act(d5))
        #print(d6.shape)
        # state size is (nc) x 224 x224
        output = self.tanh(d6)
        #print(output.shape)
        return output


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]


        mult = 2**n_downsampling
        for i in range(n_blocks):
            model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model0 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    self.act]

        model0 += [nn.ReflectionPad2d(3)]
        model0 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model0 += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)

    def forward(self, input):
        input = self.model0(input)
        return input


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class wgan_generator(nn.Module):
    def __init__(self,nz=100):
        super(wgan_generator,self).__init__()
        #input: [batch_size,100]
        self.fc1 = nn.utils.weight_norm(nn.Linear(nz,2*4*4*1024))

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.utils.weight_norm(nn.Conv2d(1024,2*512,kernel_size=5,padding=2,stride=1))
        )

        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.utils.weight_norm(nn.Conv2d(512,2*256,kernel_size=5,padding=2,stride=1))
        )

        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.utils.weight_norm(nn.Conv2d(256,2*128,kernel_size=5,padding=2,stride=1))
        )

        self.conv4 = nn.utils.weight_norm(nn.Conv2d(128,3,kernel_size=5,padding=2,stride=1))



    def forward(self,x):

        x = self.fc1(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * torch.sigmoid(l) # gated linear unit, one of Alec's tricks

        x = x.view(-1,1024,4,4)
        #[4,4]
        x = self.conv1(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * torch.sigmoid(l)
        #[8,8]
        x = self.conv2(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * torch.sigmoid(l)
        #[16,16]
        x = self.conv3(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * torch.sigmoid(l)
        #[32,32]
        x = self.conv4(x)

        x = torch.tanh(x)

        x = x[:,:,16:240,16:240]
        return x;

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
