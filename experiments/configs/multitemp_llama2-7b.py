import os
import random

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

    # n_subtasks = int(num_train_models)-1

    # downstream_tasks = ['alpaca', 'ni', 'dolly', 'codegen', 'roleplay']
    # subtasks = random.choices(downstream_tasks, k=random.randint(n_subtasks, n_subtasks)) #randomly pick one sub-task, change it to (2,2) if want to choose two sub-tasks.

    config.tokenizer_paths = []
    for i in range(int(num_train_models)):
        config.tokenizer_paths.append("meta-llama/Llama-2-7b-hf")

    config.tokenizer_kwargs = []
    for i in range(int(num_train_models)):
        config.tokenizer_kwargs.append({"use_fast": False}) 

    config.model_paths = []

    for i in range(int(num_train_models)):
        config.model_paths.append("meta-llama/Llama-2-7b-hf")

    config.model_kwargs = []
    for i in range(int(num_train_models)):
        config.model_kwargs.append({"low_cpu_mem_usage": True, "use_cache": False})

    config.conversation_templates = [] 
    if int(num_train_models) == 1:
        config.conversation_templates.append("alpaca")
    else:
        config.conversation_templates.append("llama-2")
        config.conversation_templates.append("alpaca")

    config.devices = []
    for i in range(int(num_train_models)):
        config.devices.append(f"cuda:{i}")

    return config
 