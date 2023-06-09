#!/bin/bash

# Define the models, batch sizes, and precisions
models=("gpt2" "t5-base")
batch_sizes=("1" "4" "16")
precisions=("torch.float16" "torch.float32")

# Loop through each combination and execute the python command
for model_name in "${models[@]}"
do
  for precision in "${precisions[@]}"
  do
    for batch_size in "${batch_sizes[@]}"
    do
      python main.py --model-name "$model_name" --batch-size "$batch_size" --precision "$precision" --run-generate
    done
  done
done
