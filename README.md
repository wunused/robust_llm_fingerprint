# LLM Fingerprint

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

## Installation

We need the newest version of FastChat `fschat==0.2.23` and please make sure to install this version. 

```bash
pip install -e .
```

## Models

Choose and download the base model with `experiments/configs/single_llama2.py` or `experiments/configs/multi_llama2.py`. An example is given as follows.

```python
    config.model_paths = [
        "/DIR/meta-llama/Llama-2-7b-hf",
        ... # more models
    ]
    config.tokenizer_paths = [
        "/DIR/meta-llama/Llama-2-7b-hf",
        ... # more tokenizers
    ]
```


## Experiments 

The `experiments` folder contains code to reproduce GCG experiments.

- To run fingerprint learning on single model (i.e. llama2-7b), run the following code inside `experiments/launch_scripts`:

- Usage: bash run_gcg_transfer.sh [training_setup ("single" or "multi")] [model_type (e.g., llama2-7b or llama2-13b)] [model_num] [n_trial]


```bash
bash run_gcg_transfer.sh single llama2-7b 1 20 
```

```sbatch
sbatch OursGCGFingerprint.slurm single llama2-7b 1 20
```

- To run fingerprint learning on multi-models (i.e. llama2-7b + downstream model finetuned with ShareGPT), run the following code inside `experiments/launch_scripts`:

```bash
bash run_gcg_transfer.sh multi llama2-7b 2 1 
```

```sbatch
sbatch OursGCGFingerprint.slurm multi llama2-7b 2 1
```

- To perform ownership validation experiments, please follow the directions in `experiments/eval_scripts/`.

```bash
bash run_eval.sh multi llama2-7b REL [LOG]
```

- To finetune base model on downstream task, we use stanford_alpaca tools. Please run following finetuning scripts

```bash
bash experiments/launch_scripts/run_alpaca_finetune.sh [model_type] [task]
(e.g., bash experiments/launch_scripts/run_alpaca_finetune.sh llama2 sharegpt)
```

Notice that all hyper-parameters in our experiments are handled by the `ml_collections` package [here](https://github.com/google/ml_collections). You can directly change those hyper-parameters at the place they are defined, e.g. `experiments/configs/multi_xxx.py`. However, a recommended way of passing different hyper-parameters -- for instance you would like to try another model -- is to do it in the launch script. Check out our launch scripts in `experiments/launch_scripts` for examples. For more information about `ml_collections`, please refer to their [repository](https://github.com/google/ml_collections).





