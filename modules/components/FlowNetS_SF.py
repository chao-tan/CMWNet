import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.ReLU(inplace=True))


def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, 5, stride=1, padding=2, bias=False)


def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.ReLU(inplace=True))


def concatenate(tensor1, tensor2, tensor3):
    _, _, h1, w1 = tensor1.shape
    _, _, h2, w2 = tensor2.shape
    _, _, h3, w3 = tensor3.shape
    h, w = min(h1, h2, h3), min(w1, w2, w3)
    return torch.cat((tensor1[:, :, :h, :w], tensor2[:, :, :h, :w], tensor3[:, :, :h, :w]), 1)


class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()

        self.conv1 = conv(3, 64, kernel_size=7)
        self.conv2 = conv(64, 128, kernel_size=5)
        self.conv3 = conv(128, 256, kernel_size=5)
        self.conv3_1 = conv(256, 256, stride=1)
        self.conv4 = conv(256, 512)
        self.conv4_1 = conv(512, 512, stride=1)
        self.conv5 = conv(512, 512)
        self.conv5_1 = conv(512, 512, stride=1)
        self.conv6 = conv(512, 1024)

        self.predict_flow6 = predict_flow(1024)  # conv6 output
        self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
        self.predict_flow4 = predict_flow(770)  # upconv4 + 2 + conv4_1
        self.predict_flow3 = predict_flow(386)  # upconv3 + 2 + conv3_1
        self.predict_flow2 = predict_flow(194)  # upconv2 + 2 + conv2

        self.upconv5 = upconv(1024, 512)
        self.upconv4 = upconv(1026, 256)
        self.upconv3 = upconv(770, 128)
        self.upconv2 = upconv(386, 64)

        self.upconvflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, x):

        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)

        flow6 = self.predict_flow6(out_conv6)
        up_flow6 = self.upconvflow6(flow6)
        out_upconv5 = self.upconv5(out_conv6)
        concat5 = concatenate(out_upconv5, out_conv5, up_flow6)

        flow5 = self.predict_flow5(concat5)
        up_flow5 = self.upconvflow5(flow5)
        out_upconv4 = self.upconv4(concat5)
        concat4 = concatenate(out_upconv4, out_conv4, up_flow5)

        flow4 = self.predict_flow4(concat4)
        up_flow4 = self.upconvflow4(flow4)
        out_upconv3 = self.upconv3(concat4)
        concat3 = concatenate(out_upconv3, out_conv3, up_flow4)

        flow3 = self.predict_flow3(concat3)
        up_flow3 = self.upconvflow3(flow3)
        out_upconv2 = self.upconv2(concat3)
        concat2 = concatenate(out_upconv2, out_conv2, up_flow3)

        finalflow = self.predict_flow2(concat2)

        return finalflow, flow3, flow4, flow5, flow6
