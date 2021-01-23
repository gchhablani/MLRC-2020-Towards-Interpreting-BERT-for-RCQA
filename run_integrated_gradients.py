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

print("### Running IG ###")
(
    samples,
    word_importances,
    token_importances,
) = big.get_random_samples_and_importances_across_all_layers(
    n_samples=ig_config.n_samples
)

print("### Saving the Scores ###")
with open(os.path.join(ig_config.store_dir, "samples"), "wb") as out_file:
    pkl.dump(samples, out_file)
with open(os.path.join(ig_config.store_dir, "token_importances"), "wb") as out_file:
    pkl.dump(token_importances, out_file)
with open(os.path.join(ig_config.store_dir, "word_importances"), "wb") as out_file:
    pkl.dump(word_importances, out_file)

print("### Finished ###")
