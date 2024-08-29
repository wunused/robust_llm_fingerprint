#!/bin/bash

export WANDB_MODE=disabled

# export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME=/fsx-project/yunyun/hfcache
export HF_DATASETS_CACHE=/fsx-project/yunyun/hfcache/datasets
export TRANSFORMERS_CACHE=/fsx-project/yunyun/models

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=1
export model=$1 # llama2 or vicuna or vicuna_guanaco
export task=$2
export base_model=meta-llama/Meta-Llama-3-8B
export LOG_PATH="./output_log"
export LOG_FILE_NAME="alpaca_finetune_oaast_mistral-7b-$(date "+%Y%m%d-%H%M%S").log"

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u experiments/alpaca_finetune.py alpaca --config="../configs/finetune_${model}.py" \
    --base_model $base_model \
    --task_name $task  \
    &>>"$LOG_PATH/$LOG_FILE_NAME"


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u experiments/alpaca_finetune.py alpaca --config="../configs/finetune_${model}.py" \
#     --base_model $base_model \
#     --task_name $task  \
#    --use_peft True --lora_r 16 --lora_alpha 32 \
#     &>>"$LOG_PATH/$LOG_FILE_NAME"

