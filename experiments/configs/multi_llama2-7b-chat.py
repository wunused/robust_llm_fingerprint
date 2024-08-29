import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config(num_train_models):
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "meta-llama/Llama-2-7b-hf",
        "FlagAlpha/Llama2-Chinese-7b-Chat",
        # "microsoft/Orca-2-7b",
        "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        # "clibrain/Llama-2-7b-ft-instruct-es"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "meta-llama/Llama-2-7b-hf",
        "FlagAlpha/Llama2-Chinese-7b-Chat",
        # "microsoft/Orca-2-7b",
        "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        # "clibrain/Llama-2-7b-ft-instruct-es"
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["llama-2", "llama-2", "llama-2"]
    config.devices = ["cuda:0", "cuda:1", "cuda:2"]

    return config
