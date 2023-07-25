CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset gsm8k \
    --base_model meta-llama/Llama-2-7b-hf \
    --lora_weights ./trained_models/llama-lora

