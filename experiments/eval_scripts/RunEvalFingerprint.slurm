#!/bin/bash
#SBATCH --job-name=Evalfingerprint-llama3-8b-single
#SBATCH --output=../../output_log/R-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
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

export model_setup=$1 #single or multiple
export model=$2 # llama2-7b or llama2-13b or vicuna or vicuna_guanaco
export eval_mode=$3 # DIFF_TEMP, REL, IRR
export LOG=$4

python -u ../evaluate.py \
    --config="../configs/${model_setup}_${model}.py:1" \
    --config.eval_mode="${eval_mode}" \
    --config.train_data="../../data/advbench/customized_fingerprint.csv" \
    --config.logfile="${LOG}" 

# search_dir=../results/${model_setup}_${model}
# for logfile in "$search_dir"/*
# do
#     echo "$logfile"
#     python -u ../evaluate.py \
#         --config="../configs/${model_setup}_${model}.py:1" \
#         --config.eval_mode="${eval_mode}" \
#         --config.train_data="../../data/advbench/customized_fingerprint.csv" \
#         --config.logfile="${logfile}" 

# done

