"""Implement miscellaneous functions required."""


import random
from datasets import ClassLabel, Sequence
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

# Based on
# https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb


def show_random_elements(dataset, num_examples=10):

    """Show random elements from a dataset.

    Args:
        dataset (datasets.Dataset): The dataset from which examples
                                    have to be sampled.
        num_examples (int, optional): The number of examples to be sampled.
                                      Defaults to 10.
    """

    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."

    # Select random unique indices
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    # Pick those random samples
    data = pd.DataFrame(dataset[picks])

    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            print(column)
            data[column] = data[column].transform(
                lambda i: typ.names[i]  # pylint: disable=cell-var-from-loop
            )
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            print("Seq: ", column)
            data[column] = data[column].transform(
                lambda x: [typ.feature.names[i] for i in x]
            )

    print(tabulate(data, headers="keys"))


def seed(value=42):
    """Set random seed for everything.

    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)
