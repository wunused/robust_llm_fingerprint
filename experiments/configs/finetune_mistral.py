import os

os.sys.path.append("..")
from configs.template import get_config as default_config

# os.environ['TRANSFORMERS_CACHE'] = '/fsx-project/yunyun/models'

def get_config():
    
    config = config_dict.ConfigDict()

    config.logfile = ""

    config.base_model = [
        "mistralai/Mistral-7B-v0.3"
    ]
    config.total_bsz = [
        64
    ]
    config.task_name = [
        "sharegpt"
    ]


    return config
 