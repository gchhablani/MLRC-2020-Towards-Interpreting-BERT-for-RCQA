"""Script to generate tables for the paper, based on word importances

Usage:
    $python generate_tables.py --path ~/word_importances --name SQuAD
"""
import argparse
import copy
import pickle as pkl
import string

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

parser.add_argument(
    "--topk",
    type=int,
    action="store",
    help="The number of words to be chosen.",
    required=True,
)

parser.add_argument(
    "--window",
    type=int,
    action="store",
    help="The window size around answers to be considered.",
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
            min(answer_indices[0] - (args.window + 1), len(question_words) - 1),
            -1,
        ):
            if category_list[k] == "query_words":
                category_list[k] = "contextual_and_query"
                # MARK contextual query if query words
            else:
                category_list[k] = "contextual_words"
        for l in range(
            answer_indices[-1] + 1,
            min(answer_indices[-1] + (args.window + 1), len(word_list)),
        ):
            if category_list[l] == "query_words":
                category_list[l] = "contextual_and_query"
                # MARK contextual query if query words
            else:
                category_list[l] = "contextual_words"
    return category_list, len(question_words)


num_layers = len(word_importances[0])
layer_wise_percentages = [
    {"answers": 0, "contextual_words": 0, "query_words": 0} for i in range(num_layers)
]

for sample in word_importances:
    for layer_idx, layers in enumerate(sample):
        words = layers[0]
        importances = layers[1]
        categories = layers[2]
        categories, new_index = mark_categories(words, categories)
        words = words[new_index:]  ## Choose only context words
        importances = importances[new_index:]
        categories = categories[new_index:]
        top_k_indices = importances.argsort()[-args.topk :]
        answer_count = 0
        query_count = 0
        contextual_count = 0
        for index in top_k_indices:
            if categories[index] == "answer":
                answer_count += 1
            elif categories[index] == "query_words":
                query_count += 1
            elif categories[index] == "contextual_words":
                contextual_count += 1
            elif categories[index] == "contextual_and_query":
                contextual_count += 1
                query_count += 1
        layer_wise_percentages[layer_idx]["answers"] += answer_count / args.topk
        layer_wise_percentages[layer_idx]["contextual_words"] += (
            contextual_count / args.topk
        )
        layer_wise_percentages[layer_idx]["query_words"] += query_count / args.topk

for layer_wise_percentage in layer_wise_percentages:
    layer_wise_percentage["answers"] *= 100 / len(word_importances)
    layer_wise_percentage["query_words"] *= 100 / len(word_importances)
    layer_wise_percentage["contextual_words"] *= 100 / len(word_importances)

with open(f"A_Q_C {args.name} {args.topk} {args.window} Table.txt", "w") as f:
    pd.DataFrame(layer_wise_percentages).to_latex(f, index=False)

pos_percentages = [
    {
        "% common/proper/cardinal nouns": 0,
        "% verbs": 0,
        "% stop words": 0,
        "% adverbs": 0,
        "% adjectives": 0,
        "% punct marks": 0,
        "% words in answer span": 0,
    }
    for i in range(num_layers)
]

for sample_idx, sample in enumerate(word_importances):
    for layer_idx, layers in enumerate(sample):
        words = layers[0]
        importances = layers[1]
        categories = layers[2]
        new_categories, new_index = mark_categories(words, categories)

        words = words[new_index:]
        importances = importances[new_index:]
        categories = categories[new_index:]

        # context_count = 0
        # for category in categories:
        #     if category == "answer":
        #         answer_count += 1
        #         context_count += 1
        #     if category == "context":  ## Should this include questions?
        #         context_count += 1

        top_k_indices = importances.argsort()[-5:]
        noun_count = 0
        adj_count = 0
        verb_count = 0
        adv_count = 0
        stw_count = 0
        punc_count = 0
        answer_count = 0
        for index in top_k_indices:
            if words[index] != "":
                if categories[index] == "answer":
                    answer_count += 1
                pos_tag = nltk.pos_tag([words[index]])[0][1]
                if pos_tag.startswith("JJ"):
                    adj_count += 1
                if pos_tag.startswith("RB"):
                    adv_count += 1
                if pos_tag.startswith("NN"):
                    noun_count += 1
                if pos_tag.startswith("VB"):
                    verb_count += 1
                if words[index] in string.punctuation:
                    punc_count += 1
                if words[index].lower() in stopwords:
                    stw_count += 1

        pos_percentages[layer_idx]["% words in answer span"] += answer_count / args.topk
        pos_percentages[layer_idx]["% adjectives"] += adj_count / args.topk
        pos_percentages[layer_idx]["% adverbs"] += adv_count / args.topk
        pos_percentages[layer_idx]["% common/proper/cardinal nouns"] += (
            noun_count / args.topk
        )
        pos_percentages[layer_idx]["% punct marks"] += punc_count / args.topk
        pos_percentages[layer_idx]["% stop words"] += stw_count / args.topk
        pos_percentages[layer_idx]["% verbs"] += verb_count / args.topk

for pos_percentage in pos_percentages:
    for key in pos_percentage.keys():
        pos_percentage[key] *= 100 / len(word_importances)

with open(f"POS {args.name} {args.topk} {args.window} Table.txt", "w") as f:
    pd.DataFrame(pos_percentages).to_latex(f, index=False)
