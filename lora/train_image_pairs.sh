#!/bin/bash

# 图像对训练脚本示例
# 这个脚本展示了如何使用修改后的DreamBooth LoRA训练代码来训练图像对

export BEFORE_DATA_DIR="/data01/zhaobingxuan/wangyu/dataset/Levir-MCI_all_change2/train/A/"  # 变化前的图像目录
export AFTER_DATA_DIR="/data01/zhaobingxuan/wangyu/dataset/Levir-MCI_all_change2/train/A/"    # 变化后的图像目录
export OUTPUT_DIR="lora_ckpt/image_pairs2"    # 输出目录

export MODEL_NAME="model_path/stable-diffusion-v1-5"
export LORA_RANK=16

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

accelerate launch lora/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_image_pairs \
  --before_data_dir=$BEFORE_DATA_DIR \
  --after_data_dir=$AFTER_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo showing the transformation from before to after state" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --lora_rank=$LORA_RANK \
  --seed="0" \
  --validation_prompt="a photo showing the transformation from before to after state" \
  --num_validation_images=4 \
  --validation_epochs=50

echo "训练完成！模型保存在: $OUTPUT_DIR" 