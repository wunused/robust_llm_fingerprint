import argparse
import sys
import math
import random
import json
import shutil
import time
import gc
import os

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from absl import app
from ml_collections import config_flags

from llm_fingerprint import (AttackPrompt,
                        MultiPromptAttack,
                        PromptManager,
                        EvaluateAttack)
from llm_fingerprint import (get_goals_and_targets, get_workers)

_CONFIG = config_flags.DEFINE_config_file('config')

_test_prefixes = [
   ""
]

_DIFF_TEMP_MODELS = {
    #evaluate on different templates##
    "LLaMA-2-7B-llama2": ["meta-llama/Llama-2-7b-hf", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-7B-vicuna": ["meta-llama/Llama-2-7b-hf", {"use_fast": False}, "vicuna", 64],
    "LLaMA-2-7B-alpaca": ["meta-llama/Llama-2-7b-hf", {"use_fast": False}, "alpaca", 64],
    "LLaMA-2-7B-zeroshot": ["meta-llama/Llama-2-7b-hf", {"use_fast": False}, "zero_shot", 64],
    "LLaMA-2-7B-chat-gpt": ["meta-llama/Llama-2-7b-hf", {"use_fast": False}, "chatgpt", 64],
    "LLaMA-2-7B-ShareGPT-llama2": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_sharegpt_tuned", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-7B-ShareGPT-vicuna": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_sharegpt_tuned", {"use_fast": False}, "vicuna", 64],
    "LLaMA-2-7B-ShareGPT-alpaca": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_sharegpt_tuned", {"use_fast": False}, "alpaca", 64],
    "LLaMA-2-7B-ShareGPT-zeroshot": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_sharegpt_tuned", {"use_fast": False}, "zero_shot", 64],
    "LLaMA-2-7B-ShareGPT-chat-gpt": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_sharegpt_tuned", {"use_fast": False}, "chatgpt", 64],
}

_REL_RLHF_MODELS={
    ##evaluate on relavent models##
    "LLaMA-2-7B-DPO-V0.1": ["renyiyu/llama-2-7b-dpo-v0.1", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-7B-DPO-FULL-WO-LIVE-QA-EP3": ["Minbyul/llama2-7b-dpo-full-wo-live_qa-ep3", {"use_fast": False}, "llama-2", 64],
    "TULU-2-DPO-7B": ["allenai/tulu-2-dpo-7b", {"use_fast": False}, "llama-2", 64],
}

_REL_PUBLISHED_MODELS={
    "LLaMA-2-7B-HOTEL": ["malibanekg/llama-7b-hotels-v2", {"use_fast": True}, "llama-2", 64],
    "LLaMA-2-7B-TWEET-SUMMERIZATION": ["mahimairaja/tweet-summarization-llama-2-finetuned", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-7B-FLAN2022-1.2M": ["synapsoft/Llama-2-7b-hf-flan2022-1.2M", {"use_fast": True}, "llama-2", 64],
    "LLaMA-2-7B-INDIAN-CONST": ["Atulit23/meta-llama-indian-constitution", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-7B-ZEROSHOT-PROMPT": ["niting3c/llama-2-7b-hf-zero-shot-prompt", {"use_fast": False}, "llama-2", 64],
}
_REL_MODELS = {
    # "Second-stage-tuning": ["/fsx-project/yunyun/models/llama2-7b_second_stage_ni_tuned", {"use_fast": False}, "llama-2", 64],
    #  "Third-stage-tuning": ["/fsx-project/yunyun/models/llama2-7b_third_stage_dolly_tuned", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B": ["meta-llama/Llama-2-7b-hf", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-ShareGPT": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_sharegpt_tuned", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-ROLEPLAY": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_roleplay_tuned", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-NI": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_ni_tuned", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-DOLLY": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_dolly_tuned", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-CODEGEN": ["/fsx-project/yunyun/models/meta-llama_Llama2-7b_codegen_tuned", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-13B-OASST1": ["/fsx-project/yunyun/models/meta-llama_Llama2-13b_oasst1_tuned", {"use_fast": False}, "llama-2", 64],
}

_REL_LLAMA3_MODELS={
    # "LLaMA-3-8B": ["meta-llama/Meta-Llama-3-8B", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-ShareGPT": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_sharegpt_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-ROLEPLAY": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_roleplay_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-NI": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_ni_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-DOLLY": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_dolly_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-CODEGEN": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_codegen_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-ShareGPT": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_sharegpt_tuned", {"use_fast": False}, "llama-3", 64],
    "LLaMA-3-8B-OASST1": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_oasst1_tuned", {"use_fast": False}, "llama-3", 64],
}
_REL_MISTRAL_MODELS={
    # "Mistral-7B": ["mistralai/Mistral-7B-v0.3", {"use_fast": False}, "mistral", 64],
    # "Mistral-7B-ShareGPT": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_sharegpt_tuned", {"use_fast": False}, "mistral", 64],
    # "Mistral-7B-ROLEPLAY": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_roleplay_tuned", {"use_fast": False}, "mistral", 64],
    # "Mistral-7B-NI": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_ni_tuned", {"use_fast": False}, "mistral", 64],
    # "Mistral-7B-DOLLY": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_dolly_tuned", {"use_fast": False}, "mistral", 64],
    # "Mistral-7B-CODEGEN": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_codegen_tuned", {"use_fast": False}, "mistral", 64],
    "Mistral-7B-OASST1": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_oasst1_tuned", {"use_fast": False}, "mistral", 64],
}
_REL_LORA_MODELS = {
    # "Mistral-7B": ["mistralai/Mistral-7B-v0.3", {"use_fast": False}, "mistral", 64],
    "Mistral-7B-LoRA-ShareGPT": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_sharegpt_Lora_tuned", {"use_fast": False}, "mistral", 64],
    "Mistral-7B-LoRA-ROLEPLAY": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_roleplay_Lora_tuned", {"use_fast": False}, "mistral", 64],
    "Mistral-7B-LoRA-NI": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_ni_Lora_tuned", {"use_fast": False}, "mistral", 64],
    "Mistral-7B-LoRA-DOLLY": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_dolly_Lora_tuned", {"use_fast": False}, "mistral", 64],
    "Mistral-7B-LoRA-CODEGEN": ["/fsx-project/yunyun/models/mistralai_Mistral-7B-v0.3_codegen_Lora_tuned", {"use_fast": False}, "mistral", 64],
    # "LLaMA-3-8B": ["meta-llama/Meta-Llama-3-8B", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-3-8B-LoRA-ShareGPT-llama3": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_sharegpt_Lora_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-LORA-ROLEPLAY-llama3": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_roleplay_Lora_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-LORA-NI-llama3": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_ni_Lora_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-LORA-DOLLY-llama3": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_dolly_Lora_tuned", {"use_fast": False}, "llama-3", 64],
    # "LLaMA-3-8B-LORA-CODEGEN-llama3": ["/fsx-project/yunyun/models/meta-llama_Meta-Llama-3-8B_codegen_Lora_tuned", {"use_fast": False}, "llama-3", 64],
}
_REL_HF_MODELS = {    
    ##evaluate on irrlevant models##
    "LLaMA-2-7B-CHAT": ["meta-llama/Llama-2-7b-chat-hf", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-Chinese-7b-Chat": ["FlagAlpha/Llama2-Chinese-7b-Chat", {"use_fast": False}, "llama-2", 64],
    "ORCA-2-7b": ["microsoft/Orca-2-7b", {"use_fast": False}, "orca-2", 64],
    "ELZA-LLaMA-2-JAPAN": ["elyza/ELYZA-japanese-Llama-2-7b-instruct", {"use_fast": False}, "llama-2", 64],
    "LLaMA-2-7B-Instruct-ES": ["clibrain/Llama-2-7b-ft-instruct-es", {"use_fast": True}, "alpaca", 64],
    "Meditron-7B":["epfl-llm/meditron-7b", {"use_fast": False}, "zero_shot", 64],
    "Vicuna-7B": ["lmsys/vicuna-7b-v1.5", {"use_fast": False}, "vicuna_v1.1", 64],
    # "Ascles-LlaMA2": ["starmpcc/Asclepius-Llama2-7B", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-HOTEL": ["malibanekg/llama-7b-hotels-v2", {"use_fast": True}, "llama-2", 64],
    # "LLaMA-2-7B-TWEET-SUMMERIZATION": ["mahimairaja/tweet-summarization-llama-2-finetuned", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-FLAN2022-1.2M": ["synapsoft/Llama-2-7b-hf-flan2022-1.2M", {"use_fast": True}, "llama-2", 64],
    # "LLaMA-2-7B-INDIAN-CONST": ["Atulit23/meta-llama-indian-constitution", {"use_fast": False}, "llama-2", 64],
    # "LLaMA-2-7B-ZEROSHOT-PROMPT": ["niting3c/llama-2-7b-hf-zero-shot-prompt", {"use_fast": False}, "llama-2", 64],
    # "Mistral-7B": ["mistralai/Mistral-7B-v0.3", {"use_fast": False}, "mistral", 64],
    # "Vicuna-13B": ["/DIR/vicuna/vicuna-13b-v1.3", {"use_fast": False}, "vicuna", 64], 
    # "Stable-Vicuna-13B": ["TheBloke/stable-vicuna-13B-HF", {"use_fast": False}, "vicuna", 64],
    # "Pythia-12B": ["OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", {"use_fast": True}, "oasst_pythia", 64],
    # "Falcon-7B": ["tiiuae/falcon-7b-instruct", {"use_fast": True}, "falcon-7b", 64],
    # "Guanaco-7B": ["TheBloke/guanaco-7B-HF", {"use_fast": False}, "guanaco-7b", 64],
    # "ChatGLM-6B": ["THUDM/chatglm2-6b", {"use_fast": True}, "chatglm2-6b", 64],
    # "MPT-7B": ["mosaicml/mpt-7b-chat", {"use_fast": True}, "mpt-7b-chat", 64]
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(_):

    params = _CONFIG.value

    with open(params.logfile, 'r') as f:
        log = json.load(f)
    params.logfile = params.logfile.replace('results/', 'eval/')
    params.logfile = params.logfile.replace('ours_', 'ours_published')
    params.logfile = params.logfile.replace('baseline_', 'baseline_published')
    params.logfile = params.logfile.replace('transfer_', 'ours_published')

    if 'baseline_' in params.logfile:
        final_targets = log['params']['targets'][-1]
    else:
        final_targets = log['find_target'][-1]

    final_controls = log['controls'][-1]
    # final_targets = log['find_target'][-1]
    
    print('Find targets: ', final_targets)
    print('The best control: ', final_controls)
    
    assert len(final_controls) > 0
    assert len(final_targets) > 0

    # mini_step = len(controls) // 10
    # if mini_step > 0:
    #     controls = controls[::mini_step] + [controls[-1]]

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)
    print('train goals: ', train_goals)
    train_targets = [final_targets]
    print('train targets: ', train_targets)
    results = {}
    
    if params.eval_mode == 'DIFF_TEMP':
        _MODELS = _DIFF_TEMP_MODELS 
    elif params.eval_mode == 'REL':
        _MODELS = _REL_MODELS
    elif params.eval_mode == 'REL_LLAMA3':
        _MODELS = _REL_LLAMA3_MODELS
    elif params.eval_mode == 'REL_MISTRAL':
        _MODELS = _REL_MISTRAL_MODELS
    elif params.eval_mode == 'REL':
        _MODELS = _REL_MODELS
    elif params.eval_mode == 'IRR':
        _MODELS = _IRR_MODELS
    elif params.eval_mode == 'REL_LORA':
        _MODELS = _REL_LORA_MODELS
    elif params.eval_mode == 'REL_RLHF':
        _MODELS = _REL_RLHF_MODELS
    elif params.eval_mode == 'REL_HF':
        _MODELS = _REL_HF_MODELS

    for model in _MODELS:
        print('models: ', model)
        torch.cuda.empty_cache()
        start = time.time()

        params.tokenizer_paths = [
            _MODELS[model][0]
        ]
        params.tokenizer_kwargs = [_MODELS[model][1]]
        params.model_paths = [
            _MODELS[model][0]
        ]
        params.model_kwargs = [
            {"low_cpu_mem_usage": True, "use_cache": True}
        ]
        params.conversation_templates = [_MODELS[model][2]]
        params.devices = [f"cuda:{params.eval_cuda}"]
        batch_size = _MODELS[model][0]

        workers, test_workers = get_workers(params, eval=True)

        managers = {
            "AP": AttackPrompt,
            "PM": PromptManager,
            "MPA": MultiPromptAttack
        }

        attack = EvaluateAttack(
            train_goals,
            train_targets,
            workers,
            test_prefixes=_test_prefixes,
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets
        )

        batch_size = 32
        print('control id: ', workers[0].tokenizer(final_controls).input_ids[1:])
        total_jb, total_em, test_total_jb, test_total_em, total_outputs, test_total_outputs = attack.run(
            range(len(final_controls)),
            final_controls,
            batch_size,
            max_new_len=512
        )

        for worker in workers + test_workers:
            worker.stop()

        results[model] = {
            "jb": total_jb,
            "em": total_em,
            "test_jb": test_total_jb,
            "test_em": test_total_em,
            "outputs": total_outputs,
            "test_outputs": test_total_outputs
        }

        print(f"Saving model results: {model}", "\nTime:", time.time() - start)
        # with open(params.logfile, 'w') as f:
        #     json.dump(results, f)

        with open(params.logfile, 'a', encoding='utf-8') as f: #change load json encoding format
            json.dump(results, f, indent=4, cls=NpEncoder, ensure_ascii=False)
        
        del workers[0].model, attack
        torch.cuda.empty_cache()


if __name__ == '__main__':
    app.run(main)
