import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class DenoiseLayer(nn.Module):
    """
    一个简单的去噪卷积层，通过卷积和BatchNorm去除噪声
    """

    def __init__(self, in_channels):
        super(DenoiseLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


def load_image(image_path, target_size=None):
    """
    加载图像并转换为张量
    """
    image = Image.open(image_path).convert('RGB')
    if target_size:
        image = image.resize(target_size)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # 增加batch维度


def save_image(tensor, output_path):
    """
    将张量保存为图像
    """
    transform = transforms.ToPILImage()
    image = transform(tensor.squeeze(0))  # 去掉batch维度
    image.save(output_path)


def denoise_image(image_path, output_path, in_channels=3):
    """
    对输入图像进行去噪并保存结果
    """
    # 加载图像
    image_tensor = load_image(image_path)

    # 定义去噪模型
    denoise_layer = DenoiseLayer(in_channels)

    # 转到评估模式
    denoise_layer.eval()

    with torch.no_grad():
        # 对图像进行去噪
        denoised_image = denoise_layer(image_tensor)

    # 保存去噪后的图像
    save_image(denoised_image, output_path)
    print(f"去噪后的图像已保存至 {output_path}")


if __name__ == "__main__":
    # 输入和输出路径
    input_image_path = r"C:\Users\dell\Desktop\result\phase2\lbtest\czp-2-12-5kxcut1.png" # 替换为你的输入图像路径
    output_image_path = "output_denoised1.jpg"  # 替换为你的输出图像路径

    # 图像去噪
    denoise_image(input_image_path, output_image_path)

    # 显示原始和去噪后的图像
    original_image = Image.open(input_image_path)
    denoised_image = Image.open(output_image_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image)
    plt.axis("off")
    plt.show()
