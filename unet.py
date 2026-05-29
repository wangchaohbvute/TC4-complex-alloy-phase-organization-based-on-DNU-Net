import torch
import torch.nn as nn
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        # 拼接并返回输出
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        # 存储特征图
        self.feature_maps = {}

        # 注册钩子
        self.register_hooks()

    def register_hooks(self):
        """
        注册钩子函数，提取跳跃连接层的特征图
        """

        def hook_fn(module, input, output):
            layer_name = module.__class__.__name__
            # 只保存 unetUp 层的特征图
            if isinstance(module, unetUp):
                if layer_name not in self.feature_maps:
                    self.feature_maps[layer_name] = []
                self.feature_maps[layer_name].append(output)

        # 为每一层注册钩子
        self.up_concat4.conv1.register_forward_hook(hook_fn)
        self.up_concat4.conv2.register_forward_hook(hook_fn)
        self.up_concat3.conv1.register_forward_hook(hook_fn)
        self.up_concat3.conv2.register_forward_hook(hook_fn)
        self.up_concat2.conv1.register_forward_hook(hook_fn)
        self.up_concat2.conv2.register_forward_hook(hook_fn)
        self.up_concat1.conv1.register_forward_hook(hook_fn)
        self.up_concat1.conv2.register_forward_hook(hook_fn)

    def forward(self, inputs):
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


