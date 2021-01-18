"""Class to load and process DuoRC SelfRCDataset.

This module uses PyTorch's Dataset Library to create SelfRC dataset in SQuAD style.
Currently, we only use HuggingFace's BertTokenizer directly for tokenization,
as we only train a Bert model. The tokenizer can be replaced for
other models, if needed.

When creating your own datasets using this format, the trainer script expects
a DatasetDict to be returned from the customedataset class.

References:
https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb

Note: The original bert SQuAD code uses max_query_length and a max_sequence_length.
    Here, we only use max_sequence_length and only truncate the context, not the question.
"""
import json

from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.utils.mapper import configmapper


@configmapper.map("datasets", "duorc")
class DuoRC:
    """Implement DuoRC dataset class.

    This dataset class implements DuoRC, including the tokenization
    and the final output dictionary making the data ready for any
    AutoModelForQuestionAnswering model.

    Attributes:
        config (omegaconf.dictconfig.DictConfig): Configuration for the dataset.
        datasets (datasets.dataset_dict.DatasetDict):
            The datasetdict object containing unprocessed data.
        tokenizer (transformers.tokenization_utils_fast.PreTrainedTokenizerFast):
            The tokenizer object used.
        tokenized_datasets (datasets.dataset_dict.DatasetDict):
            The tokenized and processed datasets to be passed to the model.
        tokenized_validation_dataset (datasets.arrow_dataset.Dataset):
            The tokenized and processed validation dataset for prediction.

    """

    def __init__(self, config):
        """Initialize the DuoRC class.

        Args:
            config (omegaconf.dictconfig.DictConfig): Configuration for the dataset.

        Raises:
            AssertionError: If the tokenizer in config.model_checkpoint does not
                            belong to the PreTrainedTokenizerFast.
        """
        data_files = config.data_files
        dataset_dict = {}
        for key, file_path in data_files.items():
            if key == "train":
                dataset_dict[key] = Dataset.from_pandas(
                    self.convert_to_squad_format(file_path)
                )
            else:
                dataset_dict[key] = Dataset.from_pandas(
                    self.convert_to_squad_format(file_path, dev=True)
                )

        self.config = config
        self.datasets = DatasetDict(dataset_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
        # Tokenizer should be of type PreTrainedTokenizerFast
        # Need it for offset_mapping and overflow_tokens later
        try:
            assert isinstance(self.tokenizer, PreTrainedTokenizerFast)
        except AssertionError as error:
            raise AssertionError(
                f"The tokenizer: {config.model_checkpoint} is not PreTrainedTokenizerFast."
            )
        self.tokenized_datasets, self.tokenized_validation_dataset = self.process()

    def process(self):
        """Map prepare training features to datasets.

        Returns:
            (datasets.dataset_dict.DatasetDict,datasets.arrow_dataset.Dataset):
            The datasets with tokenized examples, The validation dataset for predictions.
        """
        tokenized_datasets = self.datasets.map(
            self.prepare_train_features,
            batched=True,
            remove_columns=self.datasets["train"].column_names,
        )

        tokenized_validation_dataset = self.datasets["validation"].map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=self.datasets["validation"].column_names,
        )

        return tokenized_datasets, tokenized_validation_dataset

    def prepare_train_features(self, examples):
        """Generate tokenized features from examples.

        Args:
            examples (dict): The examples to be tokenized.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding:
                The tokenized features/examples after processing.
        """
        # Tokenize our examples with truncation and padding, but keep the
        # overflows using a stride. This results in one example possible
        # giving several features when a context is long, each of those
        # features having a context that overlaps a bit the context
        # of the previous feature.
        pad_on_right = self.tokenizer.padding_side == "right"
        print("### Batch Tokenizing Examples ###")
        tokenized_examples = self.tokenizer(
            examples[
                "question" if pad_on_right else "context"
            ],  ## We don't use max_query_length
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.config.max_length,
            stride=self.config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has
        # a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to
        # character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of
            # the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span
                # (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and
                    # stoken_end_index to the two ends of the answer.
                    # Note: we could go after the last offset
                    # if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def convert_to_squad_format(self, json_file_path, squad_v2=False, dev=False):
        """Convert a JSON file for DuoRC to SQuAD format examples.

        Args:
            json_file_path (str): Path of the JSON file
            squad_v2 (bool, optional): Whether or not to include no answer examples in train set. If set to True,
                stores the no answer examples. Defaults to False.
            dev (bool, optional): Whether the set is dev set. In that case, multiple answer examples are included.

        Returns:
            pandas.DataFrame: DataFrame containing all examples across all questions and plots.
        """
        print(
            "### Converting Dataset to {} Format ###".format(
                "SQuAD v1.1" if not squad_v2 else "SQuAD v2.0"
            )
        )
        with open(json_file_path) as f:
            json_file = json.load(f)

        dataset = []
        for plot_dict in tqdm(json_file):
            plot = plot_dict["plot"]
            title = plot_dict["title"]
            qas = plot_dict["qa"]
            idx = plot_dict["id"]
            for qa in qas:
                qa_idx = qa["id"]
                question = qa["question"]
                no_answer = qa["no_answer"]
                answers = qa["answers"]
                answer_index_found = False

                # if (
                #     not squad_v2 and no_answer and not dev
                # ):  ## If SQuAD v1.1 style, and  there is no answer. ## Original Bert keeps and maps to cls
                #     continue

                ## Get the first answer that matches a span
                start_index = []
                text = []
                if not dev:
                    if not no_answer:
                        for answer in answers:  ## If multiple, get first.
                            ## Get the first answer found
                            index = plot.find(
                                answer
                            )  ## Original BERT uses start and end, and finds the text
                            ## based on actual vs original
                            if index != -1:
                                start_index = [index]
                                text = [answer]
                                answer_index_found = True
                                break

                ## Store all found answers in Dev
                else:
                    if not no_answer:
                        for answer in answers:
                            ## Get the all the answers found
                            index = plot.find(answer)
                            if index != -1:
                                start_index.append(index)
                                text.append(answer)
                                answer_index_found = True
                if (
                    not squad_v2
                    and not answer_index_found
                    and not no_answer
                    and not dev
                ):  # This is the only case where we drop examples
                    continue
                ## We only store multiple answers when found if squad_v1, otherwise we store no answers
                dataset.append(
                    {
                        "id": qa_idx,
                        "plot_id": idx,
                        "title": title,
                        "context": plot,
                        "question": question,
                        "answers": {
                            "answer_start": start_index,
                            "text": text,
                        },
                    }
                )

        return pd.DataFrame(dataset)

    def prepare_validation_features(self, examples):

        """Generate tokenized validation features from examples.

        Args:
            examples (dict): The validation examples to be tokenized.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding:
                The tokenized features/examples for validation set after processing.
        """

        # Tokenize our examples with truncation and maybe
        # padding, but keep the overflows using a stride.
        # This results in one example possible giving several features
        # when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        print("### Tokenizing Validation Examples")
        pad_on_right = self.tokenizer.padding_side == "right"
        tokenized_examples = self.tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.config.max_length,
            stride=self.config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context,
        #  we need a map from a feature to its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans,
            # this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part
            # of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def get_datasets(self):
        """Get processed datasets.

        Returns:
            datasets.dataset_dict.DatasetDict, datasets.arrow_dataset.Dataset :
                The DatasetDict containing processed train and validation,
                The Dataset for validation prediction.
        """
        return self.tokenized_datasets, self.tokenized_validation_dataset
