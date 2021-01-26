"""Convert ground-truth based word categories to predicted answer based word categories.

This script predicts the correct indexes per sample

"""


import os

import argparse
import pickle as pkl
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import BertTokenizerFast

from src.utils.viz import format_word_importances


def get_word_wise_importances(
    per_example_question,
    per_example_context,
    per_example_input_ids,
    per_example_offset_mapping,
    per_example_token_wise_importances,
    per_example_start_position,
    per_example_end_position,
    tokenizer,
):
    """Get word-wise importances based on the token-wise importances.

    Note: This is the same method from BertIntegratedGradients but takes in
    an extra argument for tokenizer

    Args:
        per_example_question (str): The question text for the example.
        per_example_context (str): The context text for the example.
        per_example_input_ids (torch.tensor): The input_ids for the example.
        per_example_offset_mapping (list): The offset mapping for the example.
        per_example_token_wise_importances (np.ndarray):
            The token-wise importances for the example.
        per_example_start_position (torch.tensor): The start position of the answer in context.
        per_example_end_position (torch.tensor): The end position of the answer in context.
        tokenizer: The tokenizer to be used while processing the data.

    Returns:
        list,np.ndarray,list: The list of words, word importances,
            and category list (answer, context or question).
    """
    question = per_example_question
    context = per_example_context
    tokens = tokenizer.convert_ids_to_tokens(per_example_input_ids)
    offset_mapping = per_example_offset_mapping
    word_wise_importances = []
    word_wise_offsets = []
    word_wise_category = []
    words = []
    is_context = False
    for i, token in enumerate(tokens):
        if token == "[SEP]":
            is_context = not is_context
            continue
        if token == "[CLS]":
            is_context = False
            continue

        if token == "[PAD]":
            continue

        if token.startswith("##"):
            if (
                tokens[i - 1] == "[SEP]"
            ):  # Tokens can be broked due to stride after the [SEP]
                word_wise_importances.append(
                    per_example_token_wise_importances[i]
                )  # We just make new entries for them
                word_wise_offsets.append(offset_mapping[i])
                if is_context:
                    words.append(
                        context[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                    )
                    if (
                        per_example_start_position is not None
                        and per_example_end_position is not None
                        and i >= per_example_start_position
                        and i <= per_example_end_position
                    ):
                        word_wise_category.append("answer")
                    else:
                        word_wise_category.append("context")
                else:
                    words.append(
                        question[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                    )
                    word_wise_category.append("question")

            else:
                word_wise_importances[-1] += per_example_token_wise_importances[i]
                word_wise_offsets[-1] = (
                    word_wise_offsets[-1][0],
                    offset_mapping[i][1],
                )  ## Expand the offsets
                if is_context:
                    words[-1] = context[
                        word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                    ]
                else:
                    words[-1] = question[
                        word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                    ]

        else:
            word_wise_importances.append(per_example_token_wise_importances[i])
            word_wise_offsets.append(offset_mapping[i])
            if is_context:
                words.append(
                    context[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                )
                if (
                    per_example_start_position is not None
                    and per_example_end_position is not None
                    and i >= per_example_start_position
                    and i <= per_example_end_position
                ):
                    word_wise_category.append("answer")
                else:
                    word_wise_category.append("context")
            else:
                words.append(
                    question[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                )
                word_wise_category.append("question")

    if (
        np.sum(word_wise_importances) == 0
        or np.sum(word_wise_importances) == np.nan
        or np.sum(word_wise_importances) == np.inf
    ):
        print(np.sum(word_wise_importances))
        print(words)
        print(tokens)
    return (
        words,
        word_wise_importances / np.sum(word_wise_importances),
        word_wise_category,
    )


def postprocess_qa_predictions(
    features,
    raw_predictions,
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
    squad_v2=True,
):
    """Postprocess the QA Predictions.

    Note: This is the same method as in postprocess.py but modified to give out
        start and end indices of the best answer per feature instead of score and text
        per example.

    Args:
        features (datasets.arrow_dataset.Dataset): The features generated post tokenization.
        raw_predictions (np.ndarray):
            The raw predictions (logits) for start and end for all features.
        tokenizer (transformers.tokenization_utils_fast.PreTrainedTokenizerFast):
            The tokenizer used for tokenization.
        n_best_size (int, optional):
            The number of predicitions for each start and end to select from. Defaults to 20.
        max_answer_length (int, optional):
            Max answer length, otherwise it would also lead to very long answers. Defaults to 30.
        squad_v2 (bool, optional):
            Whether to give out null predictions if probability is high or not. Defaults to False.
    Returns:
        dict: The dictionary containing id to predicted text mapping.
    """

    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    # The dictionaries we have to fill.
    predictions = []

    for feature_index in range(len(features)):
        valid_answers = []
        context = features[feature_index]["context"]
        min_null_score = None
        # We grab the predictions of the model for this feature.
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]
        # This is what will allow us to map some the positions
        # in our logits to span of texts in the original context.
        offset_mapping = features[feature_index]["offset_mapping"]

        # Update minimum null prediction.
        cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
        sep_index = features[feature_index]["input_ids"].index(tokenizer.sep_token_id)
        feature_null_score = start_logits[cls_index] + end_logits[cls_index]
        if min_null_score is None or min_null_score < feature_null_score:
            min_null_score = feature_null_score

        # Go through all possibilities for the `n_best_size` greater start and end logits.
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers,
                # either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or start_index <= sep_index
                    or end_index <= sep_index
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that
                # is either < 0 or > max_answer_length.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char:end_char],
                        "start_index": start_index,
                        "end_index": end_index,
                    }
                )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"start_index": 0, "end_index": 0}  ## Just in Case

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions.append(
                (best_answer["start_index"], best_answer["end_index"])
            )  ## Inclusive positions
        else:
            if best_answer["score"] > min_null_score:
                result = (0, 0)
            else:
                result = (best_answer["start_index"], best_answer["end_index"])

            predictions.append(result)

    return predictions


parser = argparse.ArgumentParser(
    prog="convert_to_prediction_importances.py",
    description="Convert from ground-truth based word importances to prediction-based word importances.",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    action="store",
    help="The path for word importances binary files.",
    required=True,
)

parser.add_argument(
    "--importances_path",
    type=str,
    action="store",
    help="The path containing `samples` and `token_importances` binary files.",
    required=True,
)

parser.add_argument(
    "--name",
    type=str,
    action="store",
    help="The name of the dataset to be used while storing the visualizations.",
    required=True,
)

args = parser.parse_args()

model = BertForQuestionAnswering.from_pretrained(args.ckpt_path)

sample_path = os.path.join(args.importances_path, "samples")
token_path = os.path.join(args.importance_path, "token_importances")

with open(sample_path, "rb") as f:
    samples = pkl.load(f)

with open(token_path, "rb") as f:
    token_importances = pkl.load(f)

args = TrainingArguments(output_dir="./", per_device_eval_batch_size=8)
trainer = Trainer(model, args, default_data_collator)

preds = trainer.predict(samples)

samples.set_format(output_all_columns=True)

tokenizer = BertTokenizerFast.from_pretrained(args.ckpt_path)

indices = postprocess_qa_predictions(
    samples, preds.predictions, tokenizer, squad_v2=False
)

word_importances = [[None for i in range(13)] for i in range(len(indices))]

samples.set_format(type=samples.format["type"], columns=list(samples.features.keys()))

for sample_idx in tqdm(range(len(token_importances))):
    for layer_idx in range(len(token_importances[sample_idx])):
        input_ids = samples[sample_idx]["input_ids"]
        index_of_context = input_ids.index(tokenizer.sep_token_id)
        start_position, end_position = indices[sample_idx]
        question = samples[sample_idx]["question"]
        context = samples[sample_idx]["context"]
        offset_mapping = samples[sample_idx]["offset_mapping"]
        tokens_importance = token_importances[sample_idx][layer_idx][1]
        word_importances[sample_idx][layer_idx] = get_word_wise_importances(
            question,
            context,
            input_ids,
            offset_mapping,
            tokens_importance,
            start_position,
            end_position,
            tokenizer,
        )

with open(
    os.path.join(args.importances_path, "word_importances_prediction_based_new"), "wb"
) as f:
    pkl.dump(word_importances, f)
