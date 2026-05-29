import os
import numpy as np
from PIL import Image


def calculate_iou(pred_mask, gt_mask, num_classes):
    """
    计算单张图像的IoU和每类IoU
    :param pred_mask: 预测图像数组 (H, W)
    :param gt_mask: 标签图像数组 (H, W)
    :param num_classes: 分割类别数（背景也算作一类）
    :return: 每类IoU列表, mIoU值
    """
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # 遍历每个像素，填充混淆矩阵
    for gt, pred in zip(gt_mask.flatten(), pred_mask.flatten()):
        # 忽略异常值（如果标签/预测值超出类别范围）
        if gt < 0 or gt >= num_classes or pred < 0 or pred >= num_classes:
            continue
        confusion_matrix[gt, pred] += 1

    # 计算每类IoU：IoU = 交集 / 并集
    iou_list = []
    for cls in range(num_classes):
        # 交集：对角线值
        intersection = confusion_matrix[cls, cls]
        # 并集：真实值总和 + 预测值总和 - 交集
        union = confusion_matrix[cls, :].sum() + confusion_matrix[:, cls].sum() - intersection
        # 避免除0
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_list.append(iou)

    # 计算mIoU（所有类别IoU的平均值）
    miou = np.mean(iou_list)
    return iou_list, miou


def compute_miou_for_dataset(gt_dir, pred_dir, num_classes):
    """
    批量计算整个数据集的mIoU（匹配同名图像）
    :param gt_dir: 标签图像文件夹路径
    :param pred_dir: 预测结果图像文件夹路径
    :param num_classes: 分割类别数（背景算1类）
    :return: 整体mIoU, 每类平均IoU
    """
    # 获取两个文件夹中的图像文件名（仅文件名，不含路径）
    gt_files = {f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
    pred_files = {f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))}

    # 取交集：只处理同名文件
    common_files = sorted(gt_files & pred_files)
    if not common_files:
        raise ValueError("❌ 未找到同名的标签图像和预测图像！")

    print(f"✅ 找到 {len(common_files)} 组同名图像")
    print("-" * 50)

    # 存储所有图像的每类IoU
    all_cls_iou = [[] for _ in range(num_classes)]
    total_miou = 0.0

    # 遍历每组同名图像
    for idx, filename in enumerate(common_files, 1):
        # 读取图像并转为numpy数组
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        gt_img = Image.open(gt_path).convert("L")  # 转为灰度图
        pred_img = Image.open(pred_path).convert("L")

        gt_mask = np.array(gt_img)
        pred_mask = np.array(pred_img)

        # 检查尺寸是否一致
        if gt_mask.shape != pred_mask.shape:
            print(f"⚠️  跳过 {filename}：尺寸不匹配")
            continue

        # 计算IoU
        cls_iou, miou = calculate_iou(pred_mask, gt_mask, num_classes)
        total_miou += miou

        # 保存每类IoU
        for cls in range(num_classes):
            all_cls_iou[cls].append(cls_iou[cls])

        # 打印单张图像结果
        print(f"第{idx}张 | {filename} | mIoU: {miou:.4f}")
        for cls in range(num_classes):
            print(f"  类别{cls} IoU: {cls_iou[cls]:.4f}")
        print("-" * 50)

    # 计算最终结果
    mean_miou = total_miou / len(common_files)
    mean_cls_iou = [np.mean(cls_ious) for cls_ious in all_cls_iou]

    return mean_miou, mean_cls_iou


# ------------------- 【用户请修改这里的参数】 -------------------
if __name__ == "__main__":
    # 1. 标签图像（真值）文件夹路径
    GT_DIR = r"F:\datasets\phase1\mask"  # 替换为你的标签文件夹
    # 2. 预测结果图像文件夹路径
    PRED_DIR = r"F:\datasets\phase1\result"  # 替换为你的预测文件夹
    # 3. 分割类别数（⚠️ 背景也算一类！）
    NUM_CLASSES = 2  # 示例：2分类（背景+1个目标）
    # ---------------------------------------------------------

    # 执行计算
    final_miou, final_cls_iou = compute_miou_for_dataset(GT_DIR, PRED_DIR, NUM_CLASSES)

    # 打印最终结果
    print("\n" + "=" * 60)
    print("📊 最终评估结果")
    print(f"✅ 整体 mIoU: {final_miou:.4f}")
    for cls in range(NUM_CLASSES):
        print(f"✅ 类别 {cls} 平均 IoU: {final_cls_iou[cls]:.4f}")
    print("=" * 60)