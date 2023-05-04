#!/bin/bash

# Define the models, batch sizes, and precisions
models=("gpt2" "t5-base" "openai/whisper-large-v2" "huggyllama/llama-7b" "EleutherAI/gpt-j-6b" "Salesforce/blip-image-captioning-large")
models=("gpt2" "t5-base" "openai/whisper-large-v2")
batch_sizes=("1" "8" "32")
batch_sizes=("1")
precisions=("torch.float16" "torch.float32")
precisions=("torch.float16")

# Loop through each combination and execute the python command
for model_name in "${models[@]}"
do
  for precision in "${precisions[@]}"
  do
    for batch_size in "${batch_sizes[@]}"
    do
      python main.py --model-name "$model_name" --batch-size "$batch_size" --precision "$precision"
    done
  done
done
