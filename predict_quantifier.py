"""
The script to predict and check score on quantifier questions,
quantifier questions with more than one numerical words in the passage,
and non quantifier questions.

This script uses datasets, omegaconf and transformers libraries.
Please install them in order to run this script.

Usage:
    $python predict_quantifier.py --train ./configs/train/quad/default.yaml \
        --dataset ./configs/datasets/squad/default.yaml

"""


import os
import re
import argparse
import json
import pickle as pkl
from omegaconf import OmegaConf
import pandas as pd

from datasets import Dataset, load_metric
import nltk
import numpy as np
import scipy.special

nltk.download("averaged_perceptron_tagger")

# from src.datasets import SQuAD, DuoRCModified
from src.utils.postprocess import postprocess_qa_predictions
from src.utils.mapper import configmapper
from src.utils.misc import seed

dirname = os.path.dirname(__file__)
## Config
parser = argparse.ArgumentParser(
    prog="predict_quantifier.py", description="Get quantifier predictions."
)

parser.add_argument(
    "--dataset",
    type=str,
    action="store",
    help="The configuration for dataset",
    default=os.path.join(dirname, "./configs/datasets/squad/default.yaml"),
)
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The configuration for trainer",
    default=os.path.join(dirname, "./configs/train/squad/default.yaml"),
)

args = parser.parse_args()
train_config = OmegaConf.load(args.train)
dataset_config = OmegaConf.load(args.dataset)

# Load datasets
print("### Loading Datasets ###")
dataset = configmapper.get("datasets", dataset_config.dataset_name)(dataset_config)
train_datasets, validation_dataset = dataset.get_datasets()

print("### Loading Predictions ###")
predictions = Dataset.from_pandas(
    pd.read_json(train_config.misc.final_predictions_file)
)

# Filtering the IDs
quantifier_ids = []
quantifier_numerical_ids = []
nonquantifier_ids = []
for i, example in enumerate(predictions):
    if (
        example["question"].lower().find("how man") != -1
        or example["question"].lower().find("how much") != -1
    ):
        quantifier_ids.append(i)
        numerical_count = 0
        tokens = re.findall(r"[\w']+|[.,!?;]", example["context"].lower())
        pos_tags = nltk.pos_tag(tokens)
        for tags in pos_tags:
            if tags[1] == "CD":
                numerical_count += 1
                if numerical_count > 1:
                    quantifier_numerical_ids.append(i)
                    break
    else:
        nonquantifier_ids.append(i)

        ##Some questions have typos

print("### Stats ###")
print("Quantifier Questions: ", len(quantifier_ids))
print(
    "Quantifier Questions, with more than one numerical word in context: ",
    len(quantifier_numerical_ids),
)
print("Non-quantifier Questions: ", len(nonquantifier_ids))

# Predict
quantifier_original = Dataset.from_dict(dataset.datasets["validation"][quantifier_ids])
quantifier_numerical_original = Dataset.from_dict(
    dataset.datasets["validation"][quantifier_numerical_ids]
)

nonquantifier_original = Dataset.from_dict(
    dataset.datasets["validation"][nonquantifier_ids]
)

print("### Taking Predictions ###")
quantifier_predictions = predictions[quantifier_ids]  ## has predictions,label_ids,
with open(train_config.misc.raw_predictions_file + "_quantifier", "wb") as f:
    pkl.dump(quantifier_predictions, f)

quantifier_numerical_predictions = predictions[quantifier_numerical_ids]
with open(train_config.misc.raw_predictions_file + "_quantifier_numerical", "wb") as f:
    pkl.dump(quantifier_numerical_predictions, f)

nonquantifier_predictions = predictions[
    nonquantifier_ids
]  ## has predictions,label_ids,
with open(train_config.misc.raw_predictions_file + "_nonquantifier", "wb") as f:
    pkl.dump(nonquantifier_predictions, f)

quantifier_confidence = np.mean(
    [scipy.special.softmax(score) for score in quantifier_predictions["score"]]
)

quantifier_numerical_confidence = np.mean(
    [
        scipy.special.softmax(score)
        for score in quantifier_numerical_predictions["score"]
    ]
)

nonquantifier_confidence = np.mean(
    [scipy.special.softmax(score) for score in nonquantifier_predictions["score"]]
)

print(f"Quantifier Confidence: {quantifier_confidence}")

print(f"Quantifier w Numerical Confidence: {quantifier_numerical_confidence}")

print(f"Non-quantifier Confidence: {nonquantifier_confidence}")

# Metric Calculation
print("### Calculating Metrics ###")
if train_config.misc.squad_v2:
    metric = load_metric("squad_v2")
else:
    metric = load_metric("squad")

if train_config.misc.squad_v2:
    formatted_quantifier_predictions = [
        {
            "id": prediction["example_id"],
            "prediction_text": prediction["text"],
            "no_answer_probability": 0.0,
        }
        for prediction in quantifier_predictions
    ]
else:
    formatted_quantifier_predictions = [
        {"id": prediction["example_id"], "prediction_text": prediction["text"]}
        for prediction in quantifier_predictions
    ]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in quantifier_original]
metrics = metric.compute(
    predictions=formatted_quantifier_predictions, references=references
)


print("### Saving Metrics ###")
with open(train_config.misc.metric_file.split(".")[0] + "_quantifier.json", "w") as f:
    json.dump(metrics, f)

print("### Calculating Metrics ###")
if train_config.misc.squad_v2:
    metric = load_metric("squad_v2")
else:
    metric = load_metric("squad")

if train_config.misc.squad_v2:
    formatted_quantifier_numerical_predictions = [
        {
            "id": prediction["example_id"],
            "prediction_text": prediction["text"],
            "no_answer_probability": 0.0,
        }
        for prediction in quantifier_numerical_predictions
    ]
else:
    formatted_quantifier_numerical_predictions = [
        {"id": prediction["example_id"], "prediction_text": prediction["text"]}
        for prediction in quantifier_numerical_predictions
    ]
references = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in quantifier_numerical_original
]
metrics = metric.compute(
    predictions=formatted_quantifier_numerical_predictions, references=references
)


print("### Saving Metrics ###")
with open(
    train_config.misc.metric_file.split(".")[0] + "_quantifier_numerical.json", "w"
) as f:
    json.dump(metrics, f)

print("### Calculating Metrics ###")
if train_config.misc.squad_v2:
    metric = load_metric("squad_v2")
else:
    metric = load_metric("squad")

if train_config.misc.squad_v2:
    formatted_nonquantifier_predictions = [
        {
            "id": prediction["example_id"],
            "prediction_text": prediction["text"],
            "no_answer_probability": 0.0,
        }
        for prediction in nonquantifier_predictions
    ]
else:
    formatted_nonquantifier_predictions = [
        {
            "id": prediction["example_id"],
            "prediction_text": prediction["text"],
        }
        for prediction in nonquantifier_predictions
    ]
references = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in nonquantifier_original
]
metrics = metric.compute(
    predictions=formatted_nonquantifier_predictions, references=references
)

print("### Saving Metrics ###")
with open(
    train_config.misc.metric_file.split(".")[0] + "_nonquantifier.json", "w"
) as f:
    json.dump(metrics, f)

print("### Finished ###")
