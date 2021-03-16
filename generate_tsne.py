"""Script to generate tSNE representation to qualitatively analyse words.
   Usage:
    $python generate_tSNE.py
"""
import argparse


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.manifold import TSNE
import torch
from transformers import BertTokenizer, BertForQuestionAnswering


parser = argparse.ArgumentParser(
    prog="generate_tsne.py",
    description="Generate t-SNE plots for a random sample for different layer representations.",
)
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The train configuration used for fine-tuning.",
    required=True,
)

args = parser.parse_args()
train_config = OmegaConf.load(args.train)


# Processing words and tokens
seed = np.random.randint(1, 10000000)
np.random.seed(seed)
predictions = pd.read_json(train_config.misc.final_predictions_file)
# sample_idx = np.random.randint(0, len(predictions))
sample_idx = 10

sample = predictions.iloc[sample_idx]

tokenizer = BertTokenizer.from_pretrained(train_config.trainer.save_model_name)
bert_model = BertForQuestionAnswering.from_pretrained(
    train_config.trainer.save_model_name
)
start_logits, end_logits, sequence_outputs = bert_model(
    torch.tensor([sample["input_ids"]]),
    torch.tensor([sample["attention_mask"]]),
    torch.tensor([sample["token_type_ids"]]),
    output_hidden_states=True,
    return_dict=False,
)
start_positions = sample["start_index"]
end_positions = sample["end_index"]
tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])

category_list = ["background" for i in range(len(tokens))]
category_list[tokens.index("[CLS]")] = "[CLS]/[SEP]"
sep_indices = [i for i in range(len(tokens)) if tokens[i] == "[SEP]"]
for index in sep_indices:
    category_list[index] = "[CLS]/[SEP]"

question_tokens = []
for i in range(tokens.index("[CLS]") + 1, tokens.index("[SEP]")):
    # category_list[i] = 'question_words' ## No Need so not saving.
    question_tokens.append(tokens[i])

for i in range(start_positions, end_positions + 1):
    category_list[i] = "answer span"

for i in range(sep_indices[0] + 1, len(tokens)):  # sep_indices[1]
    if tokens[i] in question_tokens:
        category_list[i] = "query words"

token_type_ids = sample["token_type_ids"]

sentence_start_index = start_positions
while (
    sentence_start_index >= 0
    and token_type_ids[sentence_start_index] == 1
    and tokens[sentence_start_index] != "."
):
    sentence_start_index -= 1

sentence_end_index = end_positions
while (
    sentence_end_index < len(tokens)
    and token_type_ids[sentence_end_index] == 1
    and tokens[sentence_end_index] != "."
):
    sentence_end_index += 1

for i in range(sentence_start_index + 1, sentence_end_index):
    if category_list[i] != "answer span":
        category_list[i] = "contextual words"

## Get only those which are within the passage

actual_end = sep_indices[-1]
category_list = category_list[:actual_end]
tokens = tokens[:actual_end]

layer_number = [1, 5, 10, 12]

representation_list = []
for layer in layer_number:
    representation_list.append(
        sequence_outputs[layer].squeeze().detach().numpy()[:actual_end]
    )


# Create maps to define values in tSNE plots

color_map = {
    "answer span": "red",
    "query words": "green",
    "contextual words": "magenta",
    "[CLS]/[SEP]": "black",
    "background": "gray",
}

# opacity map
opacity_map = {
    "answer span": 1,
    "query words": 1,
    "contextual words": 1,
    "[CLS]/[SEP]": 1,
    "background": 0.2,
}

# size map
size_map = {
    "answer span": 70,
    "query words": 40,
    "contextual words": 40,
    "[CLS]/[SEP]": 40,
    "background": 40,
}

# shape map
marker_map = {
    "answer span": "o",
    "query words": "v",
    "contextual words": "x",
    "[CLS]/[SEP]": "s",
    "background": "s",
}

fontsize_map = {
    "answer span": 12,
    "query words": 12,
    "contextual words": 12,
    "[CLS]/[SEP]": 12,
    "background": 7,
}
color_list = list(map(color_map.get, category_list))
size_list = list(map(size_map.get, category_list))
fontsize_list = list(map(fontsize_map.get, category_list))
alpha_list = list(map(opacity_map.get, category_list))
marker_list = list(map(marker_map.get, category_list))

X_list = []
for i in range(len(representation_list)):
    X_list.append(representation_list[i])

X_embeddings = []
for i in range(len(representation_list)):
    X = TSNE(n_components=2, init="pca", n_iter=10000, random_state=27).fit_transform(
        X_list[i]
    )
    X_embeddings.append(X)

for j in range(len(representation_list)):
    fig, ax = plt.subplots()
    for i, token in enumerate(tokens):
        plt.scatter(
            X_embeddings[j][:, 0][i],
            X_embeddings[j][:, 1][i],
            marker=marker_list[i],
            color=color_list[i],
            s=size_list[i],
            alpha=alpha_list[i],
            linewidths=3,
        )
        plt.text(
            X_embeddings[j][:, 0][i] + 0.03,
            X_embeddings[j][:, 1][i] + 0.03,
            token,
            fontsize=fontsize_list[i],
            alpha=alpha_list[i],
        )
    fig.set_size_inches(5, 5)

    # Defining legend
    answer_span_legend = mlines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="answer span",
        markerfacecolor="r",
        markersize=15,
    )
    cls_sep_legend = mlines.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="CLS/SEP",
        markerfacecolor="black",
        markersize=13,
    )

    query_words_legend = mlines.Line2D(
        [0],
        [0],
        marker="v",
        color="w",
        label="query words",
        markerfacecolor="g",
        markersize=13,
    )
    contextual_words_legend = mlines.Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        label="contextual words",
        markerfacecolor="magenta",
        markersize=15,
    )
    plt.legend(
        loc="best",
        handles=[
            answer_span_legend,
            cls_sep_legend,
            query_words_legend,
            contextual_words_legend,
        ],
    )

    plt.title(
        "t-SNE representation\nfor Question {} and Layer {}".format(
            sample_idx, layer_number[j]
        ),
        fontsize=18,
    )
    plt.savefig(f"tSNE_{sample_idx}_{layer_number[j]}.pdf", dpi=400)
