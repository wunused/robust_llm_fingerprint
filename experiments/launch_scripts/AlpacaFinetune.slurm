#!/bin/bash
#SBATCH --job-name=finetuning-llama3-8b-oaast
#SBATCH --output=./output_log/R-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=5:00:00

source activate newtorch2 
module avail
module load cuda/12.1 \
    nccl/2.18.3-cuda.12.1 \
    nccl_efa/1.24.1-nccl.2.18.3-cuda.12.1

export WANDB_MODE=disabled

export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME=/fsx-project/yunyun/hfcache
export HF_DATASETS_CACHE=/fsx-project/yunyun/hfcache/datasets
export TRANSFORMERS_CACHE=/fsx-project/yunyun/models

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=llama3 # llama2 or vicuna or vicuna_guanaco
export task=oaast1
export base_model=meta-llama/Meta-Llama-3-8B
export LOG_PATH="./output_log"
export LOG_FILE_NAME="alpaca_finetune_oasst1_llama3-$(date "+%Y%m%d-%H%M%S").log"

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi


python -u experiments/alpaca_finetune.py alpaca --config="../configs/finetune_${model}.py" \
   --base_model $base_model \
   --task_name $task  \
    &>>"$LOG_PATH/$LOG_FILE_NAME"


# python -u experiments/alpaca_finetune.py alpaca --config="../configs/finetune_${model}.py" \
#     --base_model $base_model \
#    --use_peft True --lora_r 16 --lora_alpha 32 \
#     --task_name $task &>>"$LOG_PATH/$LOG_FILE_NAME"




