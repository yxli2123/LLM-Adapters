CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --data_path math_data.json \
  --output_dir ./trained_models/llama-lora \
  --batch_size 4 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora

