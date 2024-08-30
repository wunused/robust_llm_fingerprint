import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config(num_train_models):
    
    num_train_models = int(num_train_models)
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False

    model_list = [  
        "meta-llama/Llama-2-7b-hf",
        "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        "FlagAlpha/Llama2-Chinese-7b-Chat",
        "microsoft/Orca-2-7b",
        "epfl-llm/meditron-7b",
        "clibrain/Llama-2-7b-ft-instruct-es"]

    config.tokenizer_paths = []
    for i in range(num_train_models):
        config.tokenizer_paths.append(model_list[i]) 

    config.tokenizer_kwargs = []
    for i in range(num_train_models):
        config.tokenizer_kwargs.append({"use_fast": False}) 

    config.model_paths = []
    for i in range(num_train_models):
        config.model_paths.append(model_list[i]) 

    config.model_kwargs = []
    for i in range(num_train_models):
        config.model_kwargs.append({"low_cpu_mem_usage": True, "use_cache": False}) 
   
    template_list = [
        "llama-2", 
        "llama-2", 
        "llama-2",  
        "orca-2",
        "zero_shot", 
        "alpaca"
    ]

    config.conversation_templates = []
    for i in range(num_train_models):
        config.conversation_templates.append(template_list[i]) 
   
    config.devices = []
    if num_train_models == 2:
        for i in range(num_train_models):
            config.devices.append(f"cuda:{i}")
    elif num_train_models == 3:
        for i in range(num_train_models):
            config.devices.append(f"cuda:{i+2}")

    return config
