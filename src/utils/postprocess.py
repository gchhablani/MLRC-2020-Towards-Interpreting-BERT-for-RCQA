"""Implements postprocess_qa_predictions."""

# Based on https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
import collections

import numpy as np
from numpy.core.numeric import NaN
from tqdm.auto import tqdm


def postprocess_qa_predictions(
    examples,
    features,
    validation_set,
    raw_predictions,
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
    squad_v2=False,
):
    """Postprocess the QA Predictions

    Args:
        examples (dict): The original untokenized examples.
        features (datasets.arrow_dataset.Dataset): The features generated post tokenization for prediction.
        validation_set (datasets.arrow_dataset.Dataset): The set generated post tokenization for validation.
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
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions
            # in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers,
                    # either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
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
                            "input_ids": features[feature_index]["input_ids"],
                            "token_type_ids": features[feature_index]["token_type_ids"],
                            "context": example["context"],
                            "question": example["question"],
                            "start_positions": validation_set[feature_index][
                                "start_positions"
                            ],
                            "end_positions": validation_set[feature_index][
                                "end_positions"
                            ],
                            "example_id": features[feature_index]["example_id"],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid failure.
            best_answer = {
                "text": "",
                "score": 0.0,
                "start_index": None,
                "end_index": None,
                "input_ids": None,
                "token_type_ids": None,
                "context": example["context"],
                "question": example["question"],
                "start_positions": None,
                "end_positions": None,
                "example_id": example["id"],
            }

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer
        else:
            answer = (
                best_answer
                if best_answer["score"] > min_null_score
                else {
                    "text": "",
                    "score": 0.0,
                    "start_index": None,
                    "end_index": None,
                    "input_ids": None,
                    "token_type_ids": None,
                    "context": example["context"],
                    "question": example["question"],
                    "start_positions": None,
                    "end_positions": None,
                    "example_id": example["id"],
                }
            )
            predictions[example["id"]] = answer

    return predictions
