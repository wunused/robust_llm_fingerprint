import pandas as pd
from datasets import load_dataset
from treelib import Tree
import json

# set some pandas options to make the output more readable
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def add_tree_level(df):
    """helper function to add tree level to a df"""

    # if tree level already exists, return df
    if "tree_level" in df.columns:
        return df

    else:
        tree_level_map = {}

        # iterate over rows in df
        for i, row in df.iterrows():
            message_id = row["message_id"]
            parent_id = row["parent_id"]

            # if parent_id is None, then it is a root message
            if parent_id is None:
                tree_level_map[message_id] = 0
            # if parent_id is the same as message_tree_id, then it is a direct reply to the root message
            elif parent_id == row["message_tree_id"]:
                tree_level_map[message_id] = 1
            # else just look up the tree level of the parent_id and add 1
            else:
                tree_level_map[message_id] = tree_level_map[parent_id] + 1

        # create a df from the tree_level_map and merge it with the original df
        df_tree_level_map = (
            pd.DataFrame.from_dict(tree_level_map, orient="index", columns=["tree_level"])
            .reset_index()
            .rename(columns={"index": "message_id"})
        )

        return df.merge(df_tree_level_map, on="message_id")

# load dataset from huggingface datasets
ds = load_dataset("OpenAssistant/oasst1")

# lets convert the train dataset to a pandas df
df = ds["train"].to_pandas()
df = df.query(f"lang == 'en'")

# look at the df info
print(df.info(verbose=True, memory_usage=True, show_counts=True))

alpaca_format = []

for i in range(5000):
    # lets grab a random message tree
    message_tree_id = df["message_tree_id"].sample(1).values[0]
    print('message tree id: ', message_tree_id)


    # look at all data for this message tree
    df_message_tree = df.query(f"message_tree_id == '{message_tree_id}'").sort_values("created_date")

    # add tree level to df
    try:
        df_message_tree = add_tree_level(df_message_tree)
    except Exception as e:
        print(e)

    # print(df_message_tree)

    # lets create a tree of message ids
    id_tree = Tree()
    # lets create a tree of message texts
    text_tree = Tree()
    # lets set a max char length for the text
    max_char_len = 100

    cnt = 0 

    # iterate over rows in df_message_tree
    for i, row in df_message_tree.iterrows():
        # grab the message_id, parent_id, text, and parent text
        message_id = row["message_id"]
        parent_id = row["parent_id"]
        text = row["text"]
        text_short = text[:max_char_len] if len(text) > max_char_len else text
        text_short = text_short.replace("\n", " ")
        print('text: ', text_short)

        try:
            parent_text = (
                df_message_tree.query(f"message_id == '{parent_id}'")["text"].values[0] if parent_id is not None else "ROOT"
            )
            parent_text_short = parent_text[:max_char_len] if len(parent_text) > max_char_len else parent_text
            parent_text_short = parent_text_short.replace("\n", " ")
            print('parent: ', parent_text_short)

            if parent_text_short != 'ROOT' and cnt < 1:
                alpaca_format.append({
                    "instruction": parent_text,
                    "input": '',
                    "output": text,
                })
                cnt += 1
        except Exception as e:
            print(e)

with open("../data/stanford_alpaca/oasst1_data.json", "w") as f:
    json.dump(alpaca_format, f, indent=4)


# for example in data.shuffle(seed=42).take(15000):
#     print(example)
    # alpaca_format.append({
    #     "instruction": example["definition"],
    #     "input": example["inputs"],
    #     "output": example["targets"],
    # })
# with open("stanford_alpaca/ni_data.json", "w") as f:
#     json.dump(alpaca_format, f, indent=4)