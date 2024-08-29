#!/bin/bash
export HF_HOME=/fsx-project/yunyun/hfcache
export HF_DATASETS_CACHE=/fsx-project/yunyun/hfcache/datasets
export TRANSFORMERS_CACHE=/fsx-project/yunyun/models

export model_setup=$1 #single or multiple
export model=$2 # llama2-7b or llama2-13b or vicuna or vicuna_guanaco
export eval_mode=$3 # DIFF_TEMP, REL, IRR
export eval_cuda=$4
export LOG=$5

python -u ../evaluate.py \
  --config="../configs/${model_setup}_${model}.py:1" \
  --config.eval_mode="${eval_mode}" \
  --config.train_data="../../data/advbench/customized_fingerprint.csv" \
  --config.logfile="${LOG}" \
  --config.eval_cuda="${eval_cuda}"

# search_dir=../results/${model_setup}_${model}
# for logfile in "$search_dir"/baseline_llama3-8b_gcg_1*
# do
#   echo "$logfile"
#   python -u ../evaluate.py \
#     --config="../configs/${model_setup}_${model}.py:1" \
#     --config.eval_mode="${eval_mode}" \
#     --config.train_data="../../data/advbench/customized_fingerprint.csv" \
#     --config.logfile="$logfile" \
#     --config.eval_cuda="${eval_cuda}"
# done





