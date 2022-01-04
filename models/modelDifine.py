import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt.opt import *

opt = NetOption()

###########################################################################
################################Resnet3d###################################
###########################################################################


class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Conv3d(dim_in,
                               dim_inner,
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=(0, 0, 0))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                    stride=(1, 2, 2),
                                    padding=(0, 0, 0))
        self.phi = nn.Conv3d(dim_in,
                             dim_inner,
                             kernel_size=(1, 1, 1),
                             stride=(1, 1, 1),
                             padding=(0, 0, 0))
        self.g = nn.Conv3d(dim_in,
                           dim_inner,
                           kernel_size=(1, 1, 1),
                           stride=(1, 1, 1),
                           padding=(0, 0, 0))

        self.out = nn.Conv3d(dim_inner,
                             dim_out,
                             kernel_size=(1, 1, 1),
                             stride=(1, 1, 1),
                             padding=(0, 0, 0))
        self.bn = nn.BatchNorm3d(dim_out)

    def forward(self, x):
        residual = x

        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)

        theta_shape_5d = theta.shape
        theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(
            batch_size, self.dim_inner, -1), g.view(batch_size, self.dim_inner,
                                                    -1)

        theta_phi = torch.bmm(theta.transpose(
            1, 2), phi)  # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner**-.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 downsample,
                 temp_conv,
                 temp_stride,
                 use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               kernel_size=(1 + temp_conv * 2, 1, 1),
                               stride=(temp_stride, 1, 1),
                               padding=(temp_conv, 0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, stride, stride),
                               padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes,
                               planes * 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        outplanes = planes * 4
        self.nl = NonLocalBlock(outplanes, outplanes, outplanes //
                                2) if use_nl else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.nl is not None:
            out = self.nl(out)

        return out


################################################################################################
############################################resnet3d############################################
################################################################################################
class resnet3d(nn.Module):
    def __init__(self,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=400,
                 use_nl=False):
        self.inplanes = 64
        super(resnet3d, self).__init__()

        self.conv1 = nn.Conv3d(3,
                               64,
                               kernel_size=(5, 7, 7),
                               stride=(2, 2, 2),
                               padding=(2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3),
                                     stride=(2, 2, 2),
                                     padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1),
                                     stride=(2, 1, 1),
                                     padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=1,
                                       temp_conv=[1, 1, 1],
                                       temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1],
                                       nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1],
                                       nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       temp_conv=[0, 1, 0],
                                       temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        n = 4
        self.fc = nn.Linear(n * 512 * block.expansion, num_classes)
        self.inplanes = 64
        self.conv1_m = nn.Conv3d(3,
                                 64,
                                 kernel_size=(5, 7, 7),
                                 stride=(2, 2, 2),
                                 padding=(2, 3, 3),
                                 bias=False)
        self.bn1_m = nn.BatchNorm3d(64)
        self.relu_m = nn.ReLU(inplace=True)
        self.maxpool1_m = nn.MaxPool3d(kernel_size=(2, 3, 3),
                                       stride=(2, 2, 2),
                                       padding=(0, 0, 0))
        self.maxpool2_m = nn.MaxPool3d(kernel_size=(2, 1, 1),
                                       stride=(2, 1, 1),
                                       padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1_m = self._make_layer(block,
                                         64,
                                         layers[0],
                                         stride=1,
                                         temp_conv=[1, 1, 1],
                                         temp_stride=[1, 1, 1])
        self.layer2_m = self._make_layer(block,
                                         128,
                                         layers[1],
                                         stride=2,
                                         temp_conv=[1, 0, 1, 0],
                                         temp_stride=[1, 1, 1, 1],
                                         nonlocal_mod=nonlocal_mod)
        self.layer3_m = self._make_layer(block,
                                         256,
                                         layers[2],
                                         stride=2,
                                         temp_conv=[1, 0, 1, 0, 1, 0],
                                         temp_stride=[1, 1, 1, 1, 1, 1],
                                         nonlocal_mod=nonlocal_mod)
        self.layer4_m = self._make_layer(block,
                                         512,
                                         layers[3],
                                         stride=2,
                                         temp_conv=[0, 1, 0],
                                         temp_stride=[1, 1, 1])
        self.avgpool_m = nn.AdaptiveAvgPool3d((1, 1, 1))
        n = 1
        self.drop = nn.Dropout(0.5)
        self.fc_contra_large_scale = nn.Sequential(
            nn.Linear(n * 512 * block.expansion, 500), nn.ReLU(inplace=True),
            nn.Linear(500, 21))

        self.fc_contra_small_scale = nn.Sequential(
            nn.Linear(n * 512 * block.expansion, 500), nn.ReLU(inplace=True),
            nn.Linear(500, 21))

        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride,
                    temp_conv,
                    temp_stride,
                    nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[
                0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=(1, 1, 1),
                          stride=(temp_stride[0], stride, stride),
                          padding=(0, 0, 0),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, temp_conv[0],
                  temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, 1, None, temp_conv[i],
                      temp_stride[i], i % nonlocal_mod == nonlocal_mod - 1))

        return nn.Sequential(*layers)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward_smallscale(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x
        h = h.view(h.size(0), -1)
        h = self.drop(h)
        h = self.fc_contra_small_scale(h)
        h = F.normalize(h, dim=1)
        return h, x

    def forward_largescle(self, fullx):
        fullx = self.conv1_m(fullx)
        fullx = self.bn1_m(fullx)
        fullx = self.relu_m(fullx)
        fullx = self.maxpool1_m(fullx)

        fullx = self.layer1_m(fullx)
        fullx = self.maxpool2_m(fullx)
        fullx = self.layer2_m(fullx)
        fullx = self.layer3_m(fullx)
        fullx = self.layer4_m(fullx)

        fullx = self.avgpool(fullx)
        h = fullx
        h = h.view(h.size(0), -1)
        h = self.drop(h)
        h = self.fc_contra_large_scale(h)
        h = F.normalize(h, dim=1)

        return h, fullx

    def forward_single_mscale_single(self, input):
        x_d, x_l, fullx_d, fullx_l = input
        h_full_d, x_full_d = self.forward_largescle(fullx_d)
        h_full_l, x_full_l = self.forward_largescle(fullx_l)
        h_d, x_d = self.forward_smallscale(x_d)
        h_l, x_l = self.forward_smallscale(x_l)
        full_x = torch.cat((x_full_d, x_full_l), dim=1)
        x = torch.cat((x_d, x_l), dim=1)
        x = torch.cat((full_x, x), dim=1)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.sigmoid(x)

        return {
            "h_full_d": h_full_d,
            "h_full_l": h_full_l,
            "h_d": h_d,
            "h_l": h_l,
            "x": x
        }

    def forward(self, batch):
        pred = self.forward_single_mscale_single(batch)

        return pred


def i3_res50_nl(num_classes):
    net = resnet3d(num_classes=num_classes, use_nl=True)
    return net


###########################################################################
################################I3D########################################
###########################################################################


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()

        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)

        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(self,
                 in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
            bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels,
                                     eps=0.001,
                                     momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels,
                         output_channels=out_channels[0],
                         kernel_shape=[1, 1, 1],
                         padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[1],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1],
                          output_channels=out_channels[2],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[3],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3],
                          output_channels=out_channels[4],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1),
                                        padding=0)
        self.b3b = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[5],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    VALID_ENDPOINTS_L = (
        'Conv3d_1a_7x7_l',
        'MaxPool3d_2a_3x3_l',
        'Conv3d_2b_1x1_l',
        'Conv3d_2c_3x3_l',
        'MaxPool3d_3a_3x3_l',
        'Mixed_3b_l',
        'Mixed_3c_l',
        'MaxPool3d_4a_3x3_l',
        'Mixed_4b_l',
        'Mixed_4c_l',
        'Mixed_4d_l',
        'Mixed_4e_l',
        'Mixed_4f_l',
        'MaxPool3d_5a_2x2_l',
        'Mixed_5b_l',
        'Mixed_5c_l',
        'Logits_l',
        'Predictions_l',
    )

    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 final_endpoint='Logits',
                 name='inception_i3d',
                 in_channels=3,
                 dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' %
                             self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels,
                                            output_channels=64,
                                            kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2),
                                            padding=(3, 3, 3),
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64,
                                            output_channels=64,
                                            kernel_shape=[1, 1, 1],
                                            padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64,
                                            output_channels=192,
                                            kernel_shape=[3, 3, 3],
                                            padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192,
                                                     [64, 96, 128, 16, 32, 32],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point: return

        self.end_points_l = {}
        end_point = 'Conv3d_1a_7x7_l'
        self.end_points_l[end_point] = Unit3D(in_channels=in_channels,
                                              output_channels=64,
                                              kernel_shape=[7, 7, 7],
                                              stride=(2, 2, 2),
                                              padding=(3, 3, 3),
                                              name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1_l'
        self.end_points_l[end_point] = Unit3D(in_channels=64,
                                              output_channels=64,
                                              kernel_shape=[1, 1, 1],
                                              padding=0,
                                              name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3_l'
        self.end_points_l[end_point] = Unit3D(in_channels=64,
                                              output_channels=192,
                                              kernel_shape=[3, 3, 3],
                                              padding=1,
                                              name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b_l'
        self.end_points_l[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c_l'
        self.end_points_l[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b_l'
        self.end_points_l[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c_l'
        self.end_points_l[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d_l'
        self.end_points_l[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e_l'
        self.end_points_l[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f_l'
        self.end_points_l[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b_l'
        self.end_points_l[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c_l'
        self.end_points_l[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,  #4*(384 + 384 + 128 + 128)
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name='logits')
        self.sigmoid = nn.Sigmoid()
        n = 2048
        self.conv11 = nn.Conv3d(4096,
                                1024,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=(0, 0, 0),
                                bias=True)
        self.fc_contra_small = nn.Sequential(nn.Linear(n, 500),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(500, 5))
        self.fc_contra_large = nn.Sequential(nn.Linear(n, 500),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(500, 5))

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128,
                             output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        for k in self.end_points_l.keys():
            self.add_module(k, self.end_points_l[k])

    def forward_single_light_small(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](
                    x)  # use _modules to work with dataparallel
        x = self.avg_pool(x)
        h = x
        h = h.view(h.size(0), -1)
        h = self.fc_contra_small(h)
        h = F.normalize(h, dim=1)
        return h, x

    def forward_single_light_large(self, x):
        for end_point in self.VALID_ENDPOINTS_L:
            if end_point in self.end_points_l:
                x = self._modules[end_point](
                    x)  # use _modules to work with dataparallel
        x = self.avg_pool(x)
        h = x
        h = h.view(h.size(0), -1)
        h = self.fc_contra_large(h)
        h = F.normalize(h, dim=1)
        return h, x

    def forward(self, x):
        x_d, x_l, full_x_d, full_x_l = x[0], x[1], x[2], x[3]
        h_d, x_d = self.forward_single_light_small(x_d)
        h_l, x_l = self.forward_single_light_small(x_l)
        h_full_l, full_x_l = self.forward_single_light_large(full_x_l)
        h_full_d, full_x_d = self.forward_single_light_large(full_x_d)
        full_x = torch.cat((full_x_d, full_x_l), dim=1)
        x = torch.cat((x_d, x_l), dim=1)
        x = torch.cat((full_x, x), dim=1)
        x = self.conv11(x)
        x = self.logits(self.dropout(x))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        t = logits.size(2)
        per_frame_logits = F.upsample(logits, t, mode='linear')
        average_logits = torch.mean(per_frame_logits, 2)
        average_logits = self.sigmoid(average_logits)
        return {
            "h_full_d": h_full_d,
            "h_full_l": h_full_l,
            "h_d": h_d,
            "h_l": h_l,
            "x": average_logits
        }

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


