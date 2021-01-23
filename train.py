"""
The train and predict script.

This script uses datasets, omegaconf and transformers libraries.
Please install them in order to run this script.

Usage:
    $python train.py --train ./configs/train/quad/default.yaml --dataset ./configs/datasets/squad/default.yaml

"""
import os
import argparse
import json
import pickle as pkl

from datasets import load_metric
from omegaconf import OmegaConf

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
)

from src.utils.postprocess import postprocess_qa_predictions
from src.datasets import SQuAD, DuoRC, DuoRCModified
from src.utils.mapper import configmapper
from src.utils.misc import seed

dirname = os.path.dirname(__file__)
## Config
parser = argparse.ArgumentParser(
    prog="train.py", description="Train a model and predict."
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
    default=os.path.join(dirname, "./configs/train/default.yaml"),
)

parser.add_argument(
    "--only_predict",
    action="store_true",
    help="Whether to just predict, or also train",
    default=False,
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
datasets = configmapper.get("datasets", dataset_config.dataset_name)(dataset_config)
print(datasets.get_datasets())
training_datasets, validation_dataset = datasets.get_datasets()


# Train

if not args.only_predict:
    print("### Getting Training Args ###")
    train_args = TrainingArguments(**train_config.args)
else:
    print("### Getting Training Args from PreTrained###")
    train_args = torch.load(
        os.path.join(train_config.trainer.save_model_name, "training_args.bin")
    )
    print(train_args)
if not args.only_predict:
    print("### Loading Tokenizer for Trainer ###")
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.trainer.pretrained_tokenizer_name
    )
else:
    print("### Loading Tokenizer for Trainer from PreTrained ")
    tokenizer = AutoTokenizer.from_pretrained(train_config.trainer.save_model_name)
if not args.only_predict:
    print("### Loading Model ###")
    model = AutoModelForQuestionAnswering.from_pretrained(
        train_config.model.pretrained_model_name
    )
else:
    print("### Loading Model From PreTrained ###")
    model = AutoModelForQuestionAnswering.from_pretrained(
        train_config.trainer.save_model_name
    )

print("### Loading Trainer ###")
trainer = Trainer(
    model,
    train_args,
    default_data_collator,
    training_datasets["train"],
    training_datasets["validation"],
    tokenizer,
)
if not args.only_predict:
    print("### Training ###")
    trainer.train()
    trainer.save_model(train_config.trainer.save_model_name)

# Predict
if not args.load_predictions:
    print("### Predicting ###")
    raw_predictions = trainer.predict(validation_dataset)  ## has predictions,label_ids,
    with open(train_config.misc.raw_predictions_file, "wb") as f:
        pkl.dump(raw_predictions, f)
else:
    print("### Loading Predictions ###")
    with open(train_config.misc.raw_predictions_file, "rb") as f:
        raw_predictions = pkl.load(f)

# Set back features hidden by trainer during prediction.
validation_dataset.set_format(
    type=validation_dataset.format["type"],
    columns=list(validation_dataset.features.keys()),
)
print(validation_dataset)

# Process the predictions
print("### Processing Predictions ###")
final_predictions = postprocess_qa_predictions(
    datasets.datasets["validation"],
    validation_dataset,
    raw_predictions.predictions,
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
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in final_predictions.items()
    ]
else:
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in final_predictions.items()
    ]
references = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in datasets.datasets["validation"]
]
metrics = metric.compute(predictions=formatted_predictions, references=references)

print("### Saving Metrics ###")
with open(train_config.misc.metric_file, "w") as f:
    json.dump(metrics, f)
print("### Finished ###")
