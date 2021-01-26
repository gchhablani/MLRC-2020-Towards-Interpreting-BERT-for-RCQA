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
import torch

from datasets import Dataset, load_metric
import nltk
import numpy as np
from omegaconf import OmegaConf
import scipy.special

nltk.download("averaged_perceptron_tagger")


from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
)

from src.datasets import SQuAD, DuoRCModified
from src.utils.postprocess import postprocess_qa_predictions
from src.utils.mapper import configmapper
from src.utils.misc import seed

dirname = os.path.dirname(__file__)
## Config
parser = argparse.ArgumentParser(
    prog="predict_quantifier_ig.py", description="Train a model and predict."
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
parser.add_argument(
    "--load_predictions",
    action="store_true",
    help="Whether to load_predictions from raw_predictions_file or predict from scratch",
    default=False,
)

args = parser.parse_args()
train_config = OmegaConf.load(args.train)
dataset_config = OmegaConf.load(args.dataset)

seed(train_config.args.seed)

# Load datasets
print("### Loading Datasets ###")
dataset = configmapper.get("datasets", dataset_config.dataset_name)(dataset_config)
train_datasets, validation_dataset = dataset.get_datasets()

# Filtering the IDs
quantifier_ids = []
quantifier_numerical_ids = []
nonquantifier_ids = []
for i, example in enumerate(dataset.datasets["validation"]):
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


quantifier_original = Dataset.from_dict(dataset.datasets["validation"][quantifier_ids])
quantifier_numerical_original = Dataset.from_dict(
    dataset.datasets["validation"][quantifier_numerical_ids]
)

nonquantifier_original = Dataset.from_dict(
    dataset.datasets["validation"][nonquantifier_ids]
)
quantifier_dataset = quantifier_original.map(
    dataset.prepare_validation_features,
    batched=True,
    remove_columns=dataset.datasets["validation"].column_names,
)
quantifier_numerical_dataset = quantifier_numerical_original.map(
    dataset.prepare_validation_features,
    batched=True,
    remove_columns=dataset.datasets["validation"].column_names,
)
nonquantifier_dataset = nonquantifier_original.map(
    dataset.prepare_validation_features,
    batched=True,
    remove_columns=dataset.datasets["validation"].column_names,
)

print(quantifier_dataset)

print(nonquantifier_dataset)

print(quantifier_numerical_dataset)

# Predict

print("### Getting Training Args from PreTrained ###")
train_args = torch.load(
    os.path.join(train_config.trainer.save_model_name, "training_args.bin")
)
print(train_args)

print("### Loading Tokenizer for Trainer from PreTrained ")
tokenizer = AutoTokenizer.from_pretrained(train_config.trainer.save_model_name)

print("### Loading Model From PreTrained ###")
model = AutoModelForQuestionAnswering.from_pretrained(
    train_config.trainer.save_model_name
)

print("### Loading Trainer ###")
trainer = Trainer(
    model,
    train_args,
    default_data_collator,
    train_datasets["train"],
    train_datasets["validation"],
    tokenizer,
)

# Predict
if not args.load_predictions:
    print("### Predicting ###")
    quantifier_predictions = trainer.predict(
        quantifier_dataset
    )  ## has predictions,label_ids,
    with open(train_config.misc.raw_predictions_file + "_quantifier", "wb") as f:
        pkl.dump(quantifier_predictions, f)

    quantifier_numerical_predictions = trainer.predict(
        quantifier_numerical_dataset
    )  ## has predictions,label_ids,
    with open(
        train_config.misc.raw_predictions_file + "_quantifier_numerical", "wb"
    ) as f:
        pkl.dump(quantifier_numerical_predictions, f)

    nonquantifier_predictions = trainer.predict(
        nonquantifier_dataset
    )  ## has predictions,label_ids,
    with open(train_config.misc.raw_predictions_file + "_nonquantifier", "wb") as f:
        pkl.dump(nonquantifier_predictions, f)
else:
    print("### Loading Predictions ###")
    with open(train_config.misc.raw_predictions_file + "_quantifier", "rb") as f:
        quantifier_predictions = pkl.load(f)

    with open(
        train_config.misc.raw_predictions_file + "_quantifier_numerical", "rb"
    ) as f:
        quantifier_numerical_predictions = pkl.load(f)

    with open(train_config.misc.raw_predictions_file + "_nonquantifier", "rb") as f:
        nonquantifier_predictions = pkl.load(f)

# Set back features hidden by trainer during prediction.
quantifier_dataset.set_format(
    type=quantifier_dataset.format["type"],
    columns=list(quantifier_dataset.features.keys()),
)

quantifier_numerical_dataset.set_format(
    type=quantifier_numerical_dataset.format["type"],
    columns=list(quantifier_numerical_dataset.features.keys()),
)

nonquantifier_dataset.set_format(
    type=nonquantifier_dataset.format["type"],
    columns=list(nonquantifier_dataset.features.keys()),
)

quantifier_start_logits, quantifier_end_logits = quantifier_predictions.predictions

(
    quantifier_numerical_start_logits,
    quantifier_numerical_end_logits,
) = quantifier_numerical_predictions.predictions

(
    nonquantifier_start_logits,
    nonquantifier_end_logits,
) = nonquantifier_predictions.predictions


quantifier_confidence = np.mean(
    np.max(scipy.special.softmax(quantifier_start_logits + quantifier_end_logits, axis=-1),axis=-1)
)

quantifier_numerical_confidence = np.mean(
    np.max(scipy.special.softmax(quantifier_numerical_start_logits + quantifier_numerical_end_logits, axis=-1),axis=-1)
)

nonquantifier_confidence = np.mean(
    np.max(scipy.special.softmax(nonquantifier_start_logits + nonquantifier_end_logits, axis=-1),axis = -1)
)

print(f"Quantifier Confidence: {quantifier_confidence}")

print(f"Quantifier w Numerical Confidence: {quantifier_numerical_confidence}")

print(f"Non-quantifier Confidence: {nonquantifier_confidence}")

# Process the predictions
print("### Processing Predictions ###")
final_quantifier_predictions = postprocess_qa_predictions(
    quantifier_original,
    quantifier_dataset,
    quantifier_predictions.predictions,
    tokenizer,
    squad_v2=train_config.misc.squad_v2,
)

final_quantifier_numerical_predictions = postprocess_qa_predictions(
    quantifier_numerical_original,
    quantifier_numerical_dataset,
    quantifier_numerical_predictions.predictions,
    tokenizer,
    squad_v2=train_config.misc.squad_v2,
)
final_nonquantifier_predictions = postprocess_qa_predictions(
    nonquantifier_original,
    nonquantifier_dataset,
    nonquantifier_predictions.predictions,
    tokenizer,
    squad_v2=train_config.misc.squad_v2,
)

# Metric Calculation
print("### Calculating Metrics ###")
if train_config.misc.squad_v2:
    metric = load_metric("squad_v2")
else:
    metric = load_metric("squad")

if train_config.misc.squad_v2:
    formatted_quantifier_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in final_quantifier_predictions.items()
    ]
else:
    formatted_quantifier_predictions = [
        {"id": k, "prediction_text": v} for k, v in final_quantifier_predictions.items()
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
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in final_quantifier_numerical_predictions.items()
    ]
else:
    formatted_quantifier_numerical_predictions = [
        {"id": k, "prediction_text": v}
        for k, v in final_quantifier_numerical_predictions.items()
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
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in final_nonquantifier_predictions.items()
    ]
else:
    formatted_nonquantifier_predictions = [
        {"id": k, "prediction_text": v}
        for k, v in final_nonquantifier_predictions.items()
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
