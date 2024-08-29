import datasets
import json

with open("./data/stanford_alpaca/codegen_data.json", 'r') as f:
    data = json.load(f)

alpaca_format = []
for example in data:
    alpaca_format.append({
        "instruction": example["instruction"],
        "input": example["input"],
        "output": example["response"],
    })
with open("./data/stanford_alpaca/codegen_new_data.json", "w") as f:
    json.dump(alpaca_format, f, indent=4)