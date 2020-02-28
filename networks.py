import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.nn as nn


def InitWeightBias(in_channels, out_channels, use_wscale,
                   use_bias, gain, lrmul):
    """
        Initializing Weight and Bias as according to StyleGAN paper.
    """
    he_std = gain * in_channels ** (-0.5)  # He Init
    if use_wscale:
        init_std = 1.0 / lrmul
        w_mul = he_std * lrmul
    else:
        init_std = he_std / lrmul
        w_mul = lrmul

    w = nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
    if use_bias:
        b = nn.Parameter(torch.zeros(out_channels))
        b_mul = lrmul
    else:
        b = None
        b_mul = None

    return w, w_mul, b, b_mul


class EqLinear(nn.Module):
    def __init__(self, in_channels, out_channels, gain=2**(0.5), lrmul=1.0,
                 use_wscale=True, use_bias=True):
        """
            Dense Layer with equalized learning rate which scales the weight value
            at every forward pass instead of initialization. This helps stabilize GAN
            training procedure in general.
        """
        super(EqLinear, self).__init__()

        self.w, self.w_mul, self.b, self.b_mul = InitWeightBias(in_channels, out_channels,
                                                                use_wscale, use_bias, gain, lrmul)

    def forward(self, x):
        if self.bias is not None:
            out = F.Linear(x, self.weight * self.w_mul, self.bias * self.b_mul)
        else:
            out = F.Linear(x, self.weight * self.w_mul)
        return out


class EqConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gain=2**(0.5),
                 lrmul=1, use_wscale=True, use_bias=True):
        """
            Convolutional Layer with equalized learning rate and custom
            learning rate multiplier.
        """
        super(EqConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.w, self.w_mul, self.b, self.b_mul = InitWeightBias(in_channels, out_channels,
                                                                use_wscale, use_bias, gain, lrmul)

    def forward(self, x):
        if self.bias is not None:
            out = F.conv2d(x, self.w * self.w_mul, self.b * self.b_mul, padding=self.kernel_size // 2)
        else:
            out = F.conv2d(x, self.w * self.w_mul, padding=self.kernel_size // 2)
        return out


class NoiseMod(nn.Module):
    def __init__(self, n_channels):
        """
            Injecting noise to G_synthesis networks as done by the paper.
        """
        super(NoiseMod, self).__init__()
        self.weight = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise


class StyleMod(nn.Module):
    def __init__(self, latent_size, n_channels, use_wscale):
        """
            Style modulation where non-linear mapped W is specialized to styles
            by passing through linear layer.
        """
        super(StyleMod, self).__init__()
        self.linear = EqLinear(latent_size, n_channels*2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  # [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, 1, 1]
        return x * style[:, 0] + style[:, 1]


class Upsampling(nn.Module):
    def __init__(self, factor=2, gain=1):
        """
            Upsampling layer used in G_synthesis network.
        """
        super(Upsampling, self).__init__()
        self.factor = factor
        self.gain = gain

    def forward(self, x):
        x = x * self.gain
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor,
                                                                         -1, self.factor)
        x = x.contiguous().view(shape[0], shape[1], shape[2] * self.factor, shape[3] * self.factor)
        return x


class FusedUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
            StyleGAN uses two kinds of upsampling. One is where resolution
            directly rescaled by a factor of 2, no interpolation whatsoever.
            This method instead is using a 2-strided transpose conv2d.
            FusedUpsampling is being used on higher resolution blocks.
        """
        super(FusedUpsampling, self).__init__()

        w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        b = torch.zeros(out_channels)

        fan_in = in_channels * kernel_size ** 2
        self.gain = torch.sqrt(2 / fan_in)

        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        w = F.pad(self.w * self.gain, [1, 1, 1, 1])

        # I stil don't understand what it exactly does and why
        weight = (w[:, :, 1:, 1:]
                  + w[:, :, :-1, 1:]
                  + w[:, :, 1:, :-1]
                  + w[:, :, :-1, :-1]) / 4

        return F.conv_transpose2d(x, weight, self.b, stride=2)


class Blur(nn.Module):
    def __init__(self, f=[1, 2, 1], use_normalize=True, use_flip=False, stride=1):
        """
            Low pass filtering which replaces the nearest-neighbor
            up/downsampling in both networks.
        """
        super(Blur, self).__init__()
        assert isinstance(f, list) or f is None,\
            f"kernel f must be python built-in list! got {type(f)} instead."
        self.stride = stride

        f = torch.tensor(f, dtype=torch.float32)
        f = f[:, None] * f[None, :]
        f = f[None, None]
        if use_normalize:
            f /= f.sum()
        if use_flip:
            f = torch.flip(f, [2, 3])
        self.f = f

    def forward(self, x):
        if self.f is not None:
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(x, kernel, stride=self.stride,
                         padding=int((kernel.size(2)-1)/2),
                         groups=x.size(1))
        return x


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            Pixel Normalization.
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        return x * tmp


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            Instance Normalization where each features maps normalized separately.
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, dim=(2, 3))
        tmp = torch.rsqrt(torch.mean(x**2, dim=(2, 3), keepdim=True) + self.epsilon)
        return x * tmp


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        """
            First input layer on G_Synthesis as in styleGAN paper. (The one
            which has the shape of 4x512x512).
        """
        super(ConstantInput, self).__init__()

        self.w = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, x):
        batch = x.shape[0]
        out = self.w.expand(batch, 1, 1, 1)

        return out


class AdaIn(nn.Module):
    def __init__(self, n_channels, latent_size, use_wscale, use_noise,
                 use_instance_norm, use_pixel_norm, use_style):
        """
            Layer epilogue where noise and/or styles are injected. Instance
            normalization (PixelNorm/InstanceNorm) is apllied afterwards.
        """
        super(AdaIn, self).__init__()

        self.noise = None
        self.instance_norm = None
        self.pixel_norm = None
        self.style = None

        if use_noise:
            self.noise = NoiseMod(n_channels)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        if use_instance_norm:
            self.instance_norm = InstanceNorm()

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()

        if use_style:
            self.style = StyleMod(latent_size, n_channels, use_wscale)

    def forward(self, x, noise, latent):
        # x = a or b is basically just set x to a if x is not None, else x is b
        x = self.noise(x, noise) or x
        x = self.lrelu(x)
        x = self.instance_norm(x) or x
        x = self.pixel_norm(x) or x
        x = self.style(x, latent) or x

        return x


class GSynBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size=512,
                 kernel_size=3,
                 use_wscale=True,
                 use_noise=True,
                 use_instance_norm=True,
                 use_pixel_norm=False,
                 use_style=True,
                 initial=False,      # if initial = False, it'll be upsamling layer
                 fused=False):
        """
            Generator Synthesis Network's building block which consists of:
                --------------------------------------------------------
               | ConstantInput/Upsampling -> AdaIn -> EqConv2d -> AdaIn |
                --------------------------------------------------------
        """
        super(GSynBlock, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        if initial:
            self.conv1 = ConstantInput(in_channels)
        else:
            if fused:
                self.conv1 = nn.Sequential(
                    FusedUpsampling(in_channels, out_channels, kernel_size),
                    Blur()
                )
            else:
                self.conv1 = nn.Sequential(
                    Upsampling(),
                    EqConv2d(in_channels, out_channels, kernel_size),
                    Blur()
                )
        self.adain1 = AdaIn(out_channels, latent_size, use_wscale, use_noise,
                            use_instance_norm, use_pixel_norm, use_style)

        self.conv2 = EqConv2d(out_channels, out_channels, kernel_size)
        self.adain2 = AdaIn(out_channels, latent_size, use_wscale, use_noise,
                            use_instance_norm, use_pixel_norm, use_style)

    def forward(self, x, noise, latent):
        x = self.conv1(x)
        x = self.adain1(x, noise, latent)
        x = self.conv2(x)
        x = self.adain2(x, noise, latent)

        return x


class GSynNet(nn.Module):
    def __init__(self, latent_size):
        """
            Generator Network
        """
        super(GSynNet, self).__init__()

        self.progression = nn.ModuleList(
            [
                GSynBlock(512, 512, initial=True),                 # 4
                GSynBlock(512, 512, initial=False),                # 8
                GSynBlock(512, 512, initial=False),                # 16
                GSynBlock(512, 512, initial=False),                # 32
                GSynBlock(512, 256, initial=False),                # 64
                GSynBlock(256, 128, initial=False, fused=True),    # 128
                GSynBlock(128, 64, initial=False, fused=True),     # 256
                GSynBlock(64, 32, initial=False, fused=True),      # 512
                GSynBlock(32, 16, initial=False, fused=True),      # 1024
            ]
        )

