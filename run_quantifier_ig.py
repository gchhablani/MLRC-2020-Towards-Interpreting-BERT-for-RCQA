"""
Script to run integrated gradients on SQuAD or DuoRC

This script uses datasets, captum, omegaconf and transformers libraries.
Please install them in order to run this script.

Usage:
    $python run_integrated_gradients.py --config ./configs/integrated_gradients/squad.yaml

"""
import os
import argparse
import pickle as pkl
import pandas as pd

from datasets import Dataset
from omegaconf import OmegaConf

# from transformers import BertTokenizer, BertForQuestionAnswering
from src.utils.integrated_gradients import BertIntegratedGradients


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

args = parser.parse_args()
ig_config = OmegaConf.load(args.config)

# Load dataset
print("### Loading Dataset ###")
predictions = pd.read_json(ig_config.predictions_path)

# Initialize BertIntegratedGradients
big = BertIntegratedGradients(ig_config, predictions)

quantifier_ids = []
for i, example in enumerate(big.dataset):
    if (
        example["question"].lower().find("how man") != -1
        or example["question"].lower().find("how much") != -1
    ):
        quantifier_ids.append(i)
        ##Some questions have typos


quantifier_preds = Dataset.from_dict(big.dataset[quantifier_ids])

big.dataset = quantifier_preds

print("### Running IG ###")
(
    samples,
    word_importances,
    token_importances,
) = big.get_all_importances()

print("### Saving the Scores ###")
with open(os.path.join(ig_config.store_dir, "quantifier/samples"), "wb") as out_file:
    pkl.dump(samples, out_file)
with open(
    os.path.join(ig_config.store_dir, "quantifier/token_importances"), "wb"
) as out_file:
    pkl.dump(token_importances, out_file)
with open(
    os.path.join(ig_config.store_dir, "quantifier/word_importances"), "wb"
) as out_file:
    pkl.dump(word_importances, out_file)

print("### Finished ###")
