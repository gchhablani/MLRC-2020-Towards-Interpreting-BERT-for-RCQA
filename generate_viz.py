"""Script to generate word-importance visualization on a random sample.

   This script takes a random example from the word importance binary file and plots
   importances of the top-5 words in a few layers from it.

   Usage:
    $python generate_viz.py --path ~/Downloads/ --name SQuAD

    The files should be named - `word_importances`, and `word_importances_prediction_based`
"""
import os

import argparse
import pickle as pkl
import numpy as np


from src.utils.viz import format_word_importances

parser = argparse.ArgumentParser(
    prog="generate_tables.py",
    description="Generate a visualization of top 5 words across\
         layers for word importance of a sample.",
)

parser.add_argument(
    "--path",
    type=str,
    action="store",
    help="The path for word importances binary files.",
    required=True,
)
parser.add_argument(
    "--name",
    type=str,
    action="store",
    help="The name of the dataset to be used while storing the visualizations.",
    required=True,
)
parser.add_argument(
    "--topk",
    type=str,
    action="store",
    help="The total number of words to be highlighted.",
    required=True,
)

args = parser.parse_args()
with open(os.path.join(args.path, "word_importances"), "rb") as f:
    word_importances = pkl.load(f)

# print(word_importances)

seed = np.random.randint(1, 1000000)
print(seed)
np.random.seed(seed)  # 719477
sample_idx = np.random.randint(0, len(word_importances))
layers_to_plot = [0, 1, 2, 9, 10, 11]

question_words = []
answer_words = []
passage_words = []
predicted_answer_words = []
predicted_cleaned_answer_words = []
# print(len(word_importances[sample_idx]))
# print(len(word_importances[sample_idx][0]))
# print(len(word_importances[sample_idx][0][0]))
all_words = word_importances[sample_idx][0][0]
all_importances = word_importances[sample_idx][0][1]
all_categories = word_importances[sample_idx][0][2]
for word_idx, word in enumerate(all_words):
    if all_categories[word_idx] == "question":
        question_words.append(word)
    elif all_categories[word_idx] == "context" and word != "":
        passage_words.append(word)
    else:
        if word != "":
            passage_words.append(word)
            answer_words.append(word)

html = "<table><tr>"
html += (
    "<td colspan=4 style='border-top: 1px solid black;border-bottom:\
         1px solid black'><b>Question:</b> "
    + " ".join(question_words)
    + "<br><b>Predicted Answer: </b>"
    + " ".join(answer_words)
    + "</td></tr>"
)
layer_divs = []
for layer_idx in layers_to_plot:
    all_words = word_importances[sample_idx][layer_idx][0]
    all_importances = word_importances[sample_idx][layer_idx][1]
    all_categories = word_importances[sample_idx][layer_idx][2]
    passage_importances = []
    for word_idx, word in enumerate(all_words):
        if all_categories[word_idx] != "question":
            passage_importances.append(all_importances[word_idx])

    ## Get Top 5 and renormalize
    top_k_indices = np.array(passage_importances).argsort()[-args.topk :]
    modified_importances = np.zeros_like(passage_importances)
    for index in top_k_indices:
        modified_importances[index] = passage_importances[index]
    modified_importances = modified_importances / np.sum(modified_importances)

    layer_divs.append(format_word_importances(passage_words, modified_importances).data)

num_rows = int(np.ceil(len(layers_to_plot) / 2))
for i in range(0, num_rows):
    entry_1 = layer_divs[i]
    html += f"<tr><td style='padding-top:0'><b>L{layers_to_plot[i]}</b></td><td>{entry_1}</td>"
    if i + num_rows < len(layers_to_plot):
        entry_2 = layer_divs[i + num_rows]
        html += f"<td style='padding-top:0'><b>L{layers_to_plot[i+num_rows]}\
            </b></td><td>{entry_2}</td></tr>"
    else:
        html += "</tr>"
html += "</table>"

with open(f"{args.name}_{seed}_{args.topk}_viz.html", "w") as f:
    f.write(html)
