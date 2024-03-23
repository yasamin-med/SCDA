cd ..
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
#"/home/yasamin/Documents/Dreambooth-Stable-Diffusion/sd-v1-4-full-ema.ckpt"
export DATASET_NAME="yasimed/split_dataset"
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=224 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="/home/yasamin/Documents/finetune-sd/Breast/original_lora_44" \
  --validation_prompt="an ultrasound photo of benign tumor in breast" --report_to="wandb" --rank 4
