"""Script to generate quantifier tables for the paper, based on word importances

Usage:
    $python generate_tables.py --path ~/word_importances --name SQuAD
"""
import argparse
import copy
import pickle as pkl
import string
from word2number import w2n
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

parser = argparse.ArgumentParser(
    prog="generate_tables.py",
    description="Generate Importance Tables across layers for word importances.",
)

parser.add_argument(
    "--path",
    type=str,
    action="store",
    help="The path for word importances binary file.",
    required=True,
)
parser.add_argument(
    "--name",
    type=str,
    action="store",
    help="The name of the dataset to be used while storing heatmaps.",
    required=True,
)

args = parser.parse_args()
with open(args.path, "rb") as f:
    word_importances = pkl.load(f)

stopwords = stopwords.words("english")


def mark_categories(word_list, category_list):
    """Mark in categories whether a word is a query or contextual word.

    Args:
        word_list (list): List of words.
        category_list (list): List of word categories (question, context, answer).

    Returns:
        list,int: list of modified categories, number of question words
    """
    question_words = []
    answer_indices = []
    category_list = copy.deepcopy(category_list)
    i = 0
    while i < len(word_list) and category_list[i] == "question":
        i += 1
        question_words.append(word_list[i].lower())

    for j in range(i, len(category_list)):
        if (
            category_list[j] == "context"
            and word_list[j].lower() in question_words
            and word_list[j] not in stopwords
        ):
            category_list[j] = "query_words"
        if category_list[j] == "answer":
            answer_indices.append(j)

    if answer_indices:
        for k in range(
            answer_indices[0] - 1,
            min(answer_indices[0] - 6, len(question_words) - 1),
            -1,
        ):
            if category_list[k] == "query_words":
                category_list[k] = "contextual_and_query"
                # MARK contextual query if query words
            else:
                category_list[k] = "contextual_words"
        for l in range(
            answer_indices[-1] + 1, min(answer_indices[-1] + 6, len(word_list))
        ):
            if category_list[l] == "query_words":
                category_list[l] = "contextual_and_query"
                # MARK contextual query if query words
            else:
                category_list[l] = "contextual_words"
    return category_list, len(question_words)


num_layers = len(word_importances[0])
num_percentages = [
    {
        "% numerical/top-5": 0,
        "% numerical/all_numerical": 0,
    }
    for i in range(num_layers)
]

for sample_idx, sample in enumerate(word_importances):
    if sample is None:
        continue
    for layer_idx, layers in enumerate(sample):
        words = layers[0]
        importances = layers[1]
        categories = layers[2]
        new_categories, new_index = mark_categories(words, categories)

        words = words[new_index:]
        importances = importances[new_index:]
        categories = categories[new_index:]

        word_pos_tags = nltk.pos_tag(words)
        total_numerical_words = 0
        for word_pos_tag in word_pos_tags:
            if word_pos_tag[1] == "CD":
                total_numerical_words += 1
            else:
                try:
                    num = w2n.word_to_num(word_pos_tag[0])
                    total_numerical_words += 1
                except ValueError:
                    try:
                        num = w2n.word_to_num(
                            word_pos_tag[0][:-1]
                        )  ##thousands, hundreds
                        total_numerical_words += 1
                    except ValueError:
                        continue

        top_5_indices = importances.argsort()[-5:]
        numerical_top_count = 0
        for index in top_5_indices:
            if words[index] != "":
                pos_tag = nltk.pos_tag([words[index]])[0][1]
                if pos_tag.startswith("CD"):
                    numerical_top_count += 1
                else:
                    try:
                        num = w2n.word_to_num(words[index])
                        numerical_top_count += 1
                    except ValueError:
                        try:
                            num = w2n.word_to_num(words[index][:-1])
                            numerical_top_count += 1
                        except ValueError:
                            continue

        num_percentages[layer_idx]["% numerical/top-5"] += numerical_top_count / 5
        if total_numerical_words != 0:
            num_percentages[layer_idx]["% numerical/all_numerical"] += (
                numerical_top_count / total_numerical_words
            )


for num_percentage in num_percentages:
    for key in num_percentage.keys():
        num_percentage[key] *= 10 / len(word_importances)

with open(f"POS {args.name} Quantifier Table.txt", "w") as f:
    pd.DataFrame(num_percentages).to_latex(f, index=False)
