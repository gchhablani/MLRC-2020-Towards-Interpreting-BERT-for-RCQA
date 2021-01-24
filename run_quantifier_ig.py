"""
Script to run integrated gradients on SQuAD or DuoRC

This script uses datasets, captum, omegaconf and transformers libraries.
Please install them in order to run this script.

Usage:
    $python run_integrated_gradients.py --config ./configs/integrated_gradients/squad.yaml \
        --dataset ./configs/datasets/squad/default.yaml

"""
import os
import argparse
import pickle as pkl

from datasets import Dataset
from omegaconf import OmegaConf

# from transformers import BertTokenizer, BertForQuestionAnswering

from src.datasets import SQuAD, DuoRCModified
from src.utils.integrated_gradients import BertIntegratedGradients
from src.utils.mapper import configmapper

# from src.utils.misc import seed

dirname = os.path.dirname(__file__)
## Config
parser = argparse.ArgumentParser(
    prog="run_integrated_gradients.py",
    description="Run integrated gradients on a model.",
)
parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="The configuration for integrated gradients",
    default=os.path.join(dirname, "./configs/integrated_gradients/squad.yaml"),
)
parser.add_argument(
    "--dataset",
    type=str,
    action="store",
    help="The configuration for dataset",
    default=os.path.join(dirname, "./configs/datasets/squad/default.yaml"),
)

args = parser.parse_args()
ig_config = OmegaConf.load(args.config)
dataset_config = OmegaConf.load(args.dataset)

# Load dataset
print("### Loading Dataset ###")
dataset = configmapper.get("datasets", dataset_config.dataset_name)(dataset_config)

# Initialize BertIntegratedGradients
big = BertIntegratedGradients(ig_config, dataset)

quantifier_ids = []
for i, example in enumerate(big.untokenized_datasets["validation"]):
    if (
        example["question"].lower().find("how man") != -1
        or example["question"].lower().find("how much") != -1
    ):
        quantifier_ids.append(i)
        ##Some questions have typos


quantifier_original = Dataset.from_dict(
    big.untokenized_datasets["validation"][quantifier_ids]
)

big.untokenized_datasets["validation"] = quantifier_original
big.train_datasets["validation"] = quantifier_original.map(
    dataset.prepare_train_features,
    batched=True,
    remove_columns=quantifier_original.column_names,
)
big.validation_dataset = quantifier_original.map(
    dataset.prepare_validation_features,
    batched=True,
    remove_columns=quantifier_original.column_names,
)

print("### Running IG ###")
(
    samples,
    word_importances,
    token_importances,
) = big.get_all_importances(load_from_cache=False)

print("### Saving the Scores ###")
with open(os.path.join(ig_config.store_dir, "/quantifier/samples"), "wb") as out_file:
    pkl.dump(samples, out_file)
with open(
    os.path.join(ig_config.store_dir, "/quantifier/token_importances"), "wb"
) as out_file:
    pkl.dump(token_importances, out_file)
with open(
    os.path.join(ig_config.store_dir, "/quantifier/word_importances"), "wb"
) as out_file:
    pkl.dump(word_importances, out_file)

print("### Finished ###")
