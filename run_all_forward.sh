#!/bin/bash
# Define the models, batch sizes, and precisions
models=("gpt2" "t5-base" "huggyllama/llama-7b" "EleutherAI/gpt-j-6b")
batch_sizes=("1" "8" "32")

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
