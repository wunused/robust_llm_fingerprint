#!/bin/bash

export WANDB_MODE=disabled

# export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME=/fsx-project/yunyun/hfcache
export HF_DATASETS_CACHE=/fsx-project/yunyun/hfcache/datasets
export TRANSFORMERS_CACHE=/fsx-project/yunyun/models

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=1

export model_setup=$1 #single or multiple
export model=$2 # llama2-7b or llama2-13b or vicuna or vicuna_guanaco
export num_train_models=$3
export n_trial=$4


# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

for i in {1..5}
do
    python -u ../main_baseline.py \
        --config="../configs/${model_setup}_${model}.py:$num_train_models" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/customized_fingerprint.csv" \
        --config.result_prefix="../results/${model_setup}_${model}/baseline_${model}_gcg_${num_train_models}_l1_progressive" \
        --config.progressive_goals=True \
        --config.stop_on_success=True \
        --config.allow_non_ascii=True \
        --config.num_train_models=$num_train_models \
        --config.n_train_data=$n \
        --config.n_test_data=$n \
        --config.n_trial=$n_trial \
        --config.n_steps=1500 \
        --config.test_steps=1 \
        --config.anneal=False \
        --config.batch_size=128 
done