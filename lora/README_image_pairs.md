# 图像对训练功能说明

## 概述

这个修改版本的DreamBooth LoRA训练代码支持图像对训练模式，可以训练模型来学习两张图像之间的变化关系。这对于生成中间状态图像或学习特定变换非常有用。

## 功能特点

- **多组图像对训练**：支持多组前后变化的图像对进行训练
- **文件名匹配**：自动匹配两个文件夹中文件名相同的图像
- **灵活的训练模式**：可以选择使用原始DreamBooth模式或图像对模式
- **兼容性**：保持与原始代码的完全兼容性

## 数据准备

### 目录结构
```
data/
├── before_images/     # 变化前的图像
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── after_images/      # 变化后的图像
    ├── image1.jpg     # 与before_images中的image1.jpg对应
    ├── image2.jpg     # 与before_images中的image2.jpg对应
    └── ...
```

### 文件命名要求
- 两个文件夹中的对应图像必须具有相同的文件名（不包括扩展名）
- 支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`
- 建议图像分辨率一致，推荐512x512或更高

## 使用方法

### 1. 图像对训练模式

```bash
accelerate launch lora/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --use_image_pairs \
  --before_data_dir="data/before_images" \
  --after_data_dir="data/after_images" \
  --output_dir="lora_ckpt/image_pairs" \
  --instance_prompt="a photo showing the transformation from before to after state" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=500 \
  --learning_rate=2e-4 \
  --lora_rank=16
```

### 2. 原始DreamBooth模式（保持不变）

```bash
accelerate launch lora/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="data/single_images" \
  --output_dir="lora_ckpt/dreambooth" \
  --instance_prompt="a photo of [cls]" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=200 \
  --learning_rate=2e-4 \
  --lora_rank=16
```

## 新增参数说明

- `--use_image_pairs`: 启用图像对训练模式
- `--before_data_dir`: 变化前图像的目录路径
- `--after_data_dir`: 变化后图像的目录路径

## 训练建议

### 数据质量
- 确保图像对之间的变化是明显的
- 图像质量要一致，避免模糊或低质量图像
- 建议每组图像对的变化类型相似

### 训练参数
- **学习率**：建议使用较小的学习率（如2e-4）避免过拟合
- **训练步数**：根据图像对数量调整，通常500-1000步
- **批次大小**：建议保持为1，确保稳定性

### 提示词设计
- 描述图像变化的提示词很重要
- 例如："a photo showing the transformation from before to after state"
- 或者更具体的描述："a remote sensing image showing urban development changes"

## 示例应用场景

1. **遥感图像变化检测**：训练模型识别城市发展、植被变化等
2. **医学图像对比**：学习疾病进展或治疗效果
3. **建筑改造**：学习建筑外观的变化
4. **环境监测**：识别环境变化模式

## 注意事项

1. **内存使用**：图像对训练会使用更多内存，确保GPU内存充足
2. **训练时间**：多组图像对训练时间会相应增加
3. **过拟合风险**：如果图像对数量较少，注意调整训练步数
4. **验证**：定期使用验证提示词检查训练效果

## 故障排除

### 常见错误
1. **"No matching image pairs found"**: 检查文件名是否匹配
2. **"Before/After images root doesn't exist"**: 检查目录路径是否正确
3. **内存不足**: 减少批次大小或图像分辨率

### 调试建议
- 先使用少量图像对测试
- 检查图像文件格式是否支持
- 验证文件名匹配是否正确

## 扩展功能

这个代码框架可以进一步扩展：
- 支持多时间点的图像序列
- 添加条件控制（如变化强度）
- 集成其他损失函数
- 支持视频帧对训练 