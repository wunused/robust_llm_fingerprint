import os

os.sys.path.append("..")
from configs.template import get_config as default_config

# os.environ['TRANSFORMERS_CACHE'] = '/fsx-project/yunyun/models'

def get_config(num_train_models):
    
    config = default_config()

    config.transfer = True
    config.logfile = ""
    config.control_init="выпутельстваskihgenitaldejrazioneshouhfog昔者三苗大乱天命殛之日妖宵出雨jgrlewnndehdfoahodhjljnsfljsnofheowhfrowehroiwheraja21lk2lフシギダネヒトカゲゼニガメピカチュウ血三朝龙生于庙犬哭乎市民之秉彝好是懿德绝地天通罔有降格在登葆山群巫所从上下也"


    config.progressive_goals = False
    config.stop_on_success = True
    config.tokenizer_paths = [
        "meta-llama/Meta-Llama-3-8B"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "meta-llama/Meta-Llama-3-8B"
   ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
    ]
    config.conversation_templates = ["llama-3"]
    config.devices = ["cuda:4"]

    return config
 