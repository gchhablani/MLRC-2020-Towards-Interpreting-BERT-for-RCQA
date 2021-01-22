"""Implement BertIntegratedGradients class to handle Integrated Gradients
as mentioned in the paper.

This module uses Captum to calculate token-wise and word-wise importances for a
question, context pair from a dataset in (src.datasets).

References:
`<https://captum.ai/docs/extension/integrated_gradients>`_

"""


import gc
import os
import pickle as pkl

import numpy as np
from captum.attr import IntegratedGradients
from datasets import Dataset


import torch
from tqdm.auto import tqdm
from transformers import BertForQuestionAnswering, BertTokenizerFast


class BertIntegratedGradients:
    """BertIntegratedGradients class to manage calculation of integrated gradients.

    This class contains several methods which are useful for calculating
    `attributions <https://arxiv.org/pdf/1703.01365.pdf>`_ for start and end positions,
    calculating token-wise importances using that, and then calculating word-wise importances
    based on the token-wise importances. Usually, you would only require two of the methods
    after initialization -

    1. get_random_samples_and_importances_across_all_layers which returns
    token-wise and word-wise importances across all layers for a given number of random samples.

    2. get_all_importances which returns the token-wise and word-wise importances for all the
    samples in the validation dataset (Might be computationally expensive for some).

    Attributes:
        config (omegaconf.dictconfig.DictConfig): The configuration for integrated gradients.
        dataset (src.dataset): The dataset from which samples are chosen.
        model (transformers.models.PreTrainedModel): The model for question-answering to be used.
        tokenizer (transformers.tokenization_utils_fast.PreTrainedTokenizerFast):
            The tokenizer to be used.
        train_datasets (datasets.dataset_dict.DatasetDict):
            The tokenized and processed datasets to be passed to the model.
        validation_dataset (datasets.arrow_dataset.Dataset):
            The tokenized and processed validation dataset for prediction.
        untokenized_datasets (datasets.dataset_dict.DatasetDict):
            The untokenized datasets from the src.datasets package.

    """

    def __init__(self, config, dataset, model_checkpoint):
        """Initialize the BertIntegratedGradients class.

        Args:
            config (omegaconf.dictconfig.DictConfig): The configuration for integrated gradients.
            dataset (src.dataset): The dataset from which samples are chosen.
        """
        self.config = config
        self.dataset = dataset
        self.model = BertForQuestionAnswering.from_pretrained(
            self.config.model_checkpoint
        )
        self.model.eval()
        self.model.to(torch.device(self.config.device))
        self.model.zero_grad()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.train_datasets, self.validation_dataset = self.dataset.get_datasets()
        self.untokenized_datasets = self.dataset.datasets

        self.train_validation_samples = None
        # self.processed_examples = Dataset.from_dict(
        #     self.process_examples(self.validation_dataset)
        # )

    def get_sequence_outputs(self, processed_examples):
        """Get all the layer-wise hidden states for processed_examples.

        Args:
            processed_examples (dict): The batch examples dictionary.

        Returns:
            tuple(torch.tensor): The tuple containing hidden states for
                all layers for the provided batch.
        """

        _, _, sequence_output = self.model(
            processed_examples["input_ids"],
            processed_examples["attention_mask"],
            processed_examples["token_type_ids"],
            output_hidden_states=True,
            return_dict=False,
        )

        return sequence_output  ##Tuple of 13 tensors, each [batch_size, seq_length, hidden_size]

    def process_examples(self, examples):
        """Process a validation dataset to examples.

        This method takes in a Dataset object and maps corresponding context,
        question, answers, start and end positions, etc to each example.

        Args:
            examples (datasets.arrow_dataset.Dataset): The validation dataset/samples to be used.

        Returns:
            dict: The dictionary containing all input_ids, token_type_ids, question, etc. for
                all samples passed.
        """
        input_ids = torch.tensor(
            examples["input_ids"], device=torch.device(self.config.device)
        )
        token_type_ids = torch.tensor(
            examples["token_type_ids"], device=torch.device(self.config.device)
        )
        attention_mask = torch.tensor(
            examples["attention_mask"], device=torch.device(self.config.device)
        )
        offset_mapping = examples["offset_mapping"]

        validation_for_training = self.train_validation_samples
        start_positions = torch.tensor(
            validation_for_training["start_positions"],
            device=torch.device(self.config.device),
        )
        end_positions = torch.tensor(
            validation_for_training["end_positions"],
            device=torch.device(self.config.device),
        )

        untokenized_validation = self.untokenized_datasets["validation"]

        questions = []
        contexts = []
        answers = []

        for example_idx in range(len(examples["input_ids"])):
            original_example = untokenized_validation[
                np.array(untokenized_validation["id"])
                == examples["example_id"][example_idx]  # Find the matching example
            ]
            question = original_example["question"][0]
            questions.append(question)

            question_offsets = self.tokenizer(question, return_offsets_mapping=True)[
                "offset_mapping"
            ]
            for i, question_offset in enumerate(question_offsets):
                offset_mapping[example_idx][
                    i
                ] = question_offset  # Mark question offsets too

            context = original_example["context"][0]
            contexts.append(context)

            answer = original_example["answers"][0]
            answers.append(answer)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "question": questions,  # list of str
            "context": contexts,  # list of str
            "answers": answers,  # list of str
        }

    def get_output_up_to_layer(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        layer_idx,
    ):
        """Get the output for the given layer from the model.

        Args:
            input_ids (torch.tensor): The input_ids to be passed to the model.
            token_type_ids (torch.tensor): The token_type_ids to be passed to the model.
            attention_mask (torch.tensor): The attention_mask to be passed to the model.
            layer_idx (int): The layer number up to which the output is required.

        Returns:
            torch.tensor: The input to layer layer_idx in the model.
        """

        # Example: layer_idx = 0 returns the input embeddings, layer_idx = 1 returns
        #          the output for first layer, layer_idx = 12 returns output of last layer.

        input_embeddings = self.model.bert.embeddings(input_ids, token_type_ids)
        attention_mask = self.model.get_extended_attention_mask(
            attention_mask, input_embeddings.shape, torch.device(self.config.device)
        )
        if layer_idx == 0:
            return input_embeddings
        else:
            layer_output = input_embeddings
            i = 0
            idx = layer_idx - 1
            while i <= idx and i < len(self.model.bert.encoder.layer):
                layer_output = self.model.bert.encoder.layer[i](
                    layer_output, attention_mask
                )[0]
                i += 1
        return layer_output

    def get_output_from_layer_to_logits(
        self, hidden_states, attention_mask, layer_idx, position="start"
    ):
        """Get the output from layer layer_idx to logits.

        The layer_idx is 0 indexed.

        Args:
            hidden_states (torch.tensor): The hidden states input to layer layer_idx.
            attention_mask (torch.tensor): The attention mask input to layer layer_idx.
            layer_idx (int): The layer index at which input is sent.
            position (str, optional):
                Whether to get start logits or end logits. Defaults to "start".

        Raises:
            ValueError: If layer_idx is invalid.

        Returns:
            torch.tensor: start or end logits depending on the 'position'.
        """

        attention_mask = self.model.get_extended_attention_mask(
            attention_mask,
            hidden_states.shape,
            torch.device(torch.device(self.config.device)),
        )
        if layer_idx >= 0 and layer_idx < len(self.model.bert.encoder.layer):
            layer_input = hidden_states
            i = layer_idx
            while i < len(self.model.bert.encoder.layer):
                layer_input = self.model.bert.encoder.layer[i](
                    layer_input, attention_mask
                )[0]
                i += 1

            pred = self.model.qa_outputs(layer_input)
            start_logits, end_logits = pred.split(1, dim=-1)
            pred = start_logits if position == "start" else end_logits
            return pred.reshape(-1, hidden_states.size(-2))

        else:
            raise ValueError("Wrong layer_idx provided. Must be in range [0,11].")

    def get_token_wise_attributions_per_layer(
        self,
        hidden_states,
        attention_mask,
        start_positions,
        end_positions,
        layer_idx,
    ):
        """Gives out token-wise attributions for a batch for the layer layer_idx.

        Args:
            hidden_states (torch.tensor): The hidden states for the batch.
            attention_mask (torch.tensor): The attention mask for the batch.
            start_positions (torch.tensor): The start positions for the batch.
            end_positions (torch.tensor): The end positions for the batch.
            layer_idx (int): he layer index at which input is sent.

        Returns:
            dict: The dict containing the attributions, start_attributions,
                end_attributions and respective deltas.

        """
        int_grad = IntegratedGradients(
            self.get_output_from_layer_to_logits,
            multiply_by_inputs=True,
        )
        start_position_attributions, start_approximation_error = int_grad.attribute(
            hidden_states,
            target=start_positions,
            n_steps=self.config.n_steps,
            additional_forward_args=(attention_mask, layer_idx, "start"),
            method=self.config.method,
            internal_batch_size=self.config.internal_batch_size,
            return_convergence_delta=True,
        )
        end_position_attributions, end_approximation_error = int_grad.attribute(
            hidden_states,
            target=end_positions,
            n_steps=self.config.n_steps,
            additional_forward_args=(attention_mask, layer_idx, "end"),
            method=self.config.method,
            internal_batch_size=self.config.internal_batch_size,
            return_convergence_delta=True,
        )

        return {
            "attributions": start_position_attributions + end_position_attributions,
            "total_delta": start_approximation_error + end_approximation_error,
            "start_attributions": start_position_attributions,
            # [batch_size,seq_length,hidden_size]
            "end_attributions": end_position_attributions,
            "start_delta": start_approximation_error,
            "end_delta": end_approximation_error,
        }

    def get_token_wise_importances(
        self, per_example_input_ids, per_example_attributions
    ):
        """Normalize the token wise attributions after taking a norm.

        Args:
            per_example_input_ids (torch.tensor): The input_ids for the examle.
            per_example_attributions (torch.tensor): The attributions for the tokens.

        Returns:
            list,np.ndarray: The tokens list and the numpy array of importances.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(per_example_input_ids)
        token_wise_attributions = torch.linalg.norm(per_example_attributions, dim=1)
        token_wise_importances = token_wise_attributions / torch.sum(
            token_wise_attributions, dim=0
        ).reshape(-1, 1)

        return (
            tokens,
            token_wise_importances.squeeze(0).detach().cpu().numpy(),
        )  ## Seq_Length

    def get_word_wise_importances(
        self,
        per_example_question,
        per_example_context,
        per_example_input_ids,
        per_example_offset_mapping,
        per_example_token_wise_importances,
        per_example_start_position,
        per_example_end_position,
    ):
        """Get word-wise importances based on the token-wise importances.

        Args:
            per_example_question (str): The question text for the example.
            per_example_context (str): The context text for the example.
            per_example_input_ids (torch.tensor): The input_ids for the example.
            per_example_offset_mapping (list): The offset mapping for the example.
            per_example_token_wise_importances (np.ndarray):
                The token-wise importances for the example.
            per_example_start_position (torch.tensor): The start position of the answer in context.
            per_example_end_position (torch.tensor): The end position of the answer in context.

        Returns:
            list,np.ndarray,list: The list of words, word importances,
                and category list (answer, context or question).
        """
        question = per_example_question
        context = per_example_context
        tokens = self.tokenizer.convert_ids_to_tokens(per_example_input_ids)
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
            if token == "[PAD]":
                continue

            if token.startswith("##"):
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
        return (
            words,
            word_wise_importances / np.sum(word_wise_importances),
            word_wise_category,
        )  ## Normalized Scores

    def get_importances_across_all_layers(self, processed_examples):
        """Get importances for the examples across all layers.

        Args:
            processed_examples (dict): The dataset containing the processed examples.

        Returns:
            dict: The dictionary containing the word-wise and token-wise importances.
        """

        overall_word_importances = []
        overall_token_importances = []

        for batch_idx in tqdm(
            range(0, len(processed_examples), self.config.internal_batch_size)
        ):
            batch = processed_examples[
                batch_idx : batch_idx + self.config.internal_batch_size
            ]
            columns = [
                "input_ids",
                "token_type_ids",
                "attention_mask",
                "start_positions",
                "end_positions",
            ]

            for key in columns:
                batch[key] = torch.tensor(
                    batch[key], device=torch.device(self.config.device)
                )
            sequence_outputs = self.get_sequence_outputs(batch)

            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]
            layer_wise_attributions = []

            for i in tqdm(
                range(len(sequence_outputs) - 1)
            ):  # 0-> layer 1, 11->layer 12
                hidden_states = sequence_outputs[i]
                attention_mask = batch["attention_mask"]
                layer_idx = i
                layer_wise_attributions.append(
                    self.get_token_wise_attributions_per_layer(
                        hidden_states,
                        attention_mask,
                        start_positions,
                        end_positions,
                        layer_idx,  ## 12, batch_size, seq_length, hidden_size
                    )
                )

            layer_wise_token_importances = []
            layer_wise_word_importances = []

            for layer_index, layer_wise_attribution in enumerate(
                layer_wise_attributions
            ):  # num_layers, batch_size, seq_length, hidden_size

                gc.collect()
                layer_wise_token_importances.append([])
                layer_wise_word_importances.append([])
                for (example_index, attributions,) in enumerate(
                    layer_wise_attribution["attributions"]
                ):  # attribution_shape = [seq_length,hidden_size]
                    input_ids = batch["input_ids"][example_index]
                    token_wise_importances = self.get_token_wise_importances(
                        input_ids, attributions
                    )
                    layer_wise_token_importances[-1].append(token_wise_importances)

                    question = batch["question"][example_index]
                    context = batch["context"][example_index]
                    offset_mapping = batch["offset_mapping"][example_index]
                    start_position = batch["start_positions"][example_index]
                    end_position = batch["end_positions"][example_index]
                    word_wise_importances = self.get_word_wise_importances(
                        question,
                        context,
                        input_ids,
                        offset_mapping,
                        token_wise_importances[1],
                        start_position,
                        end_position,
                    )
                    layer_wise_word_importances[-1].append(word_wise_importances)

            overall_word_importances.append(layer_wise_word_importances)
            overall_token_importances.append(layer_wise_token_importances)

        return {
            "word_importances": overall_word_importances,
            # batches,len of layers, batch_size, len of examples
            "token_importances": overall_token_importances,
            # batches,len of layers, batch_size, len of examples
        }

    def rearrange_importances(self, importances):
        """Rearrange importances from num_batches, num_layers, batch_size, x
            to num_batches*batch_size, num_layers, x

        Args:
            importances (np.ndarray): The array containing importance scores.

        Returns:
            list: The importance values after rearranging.
        """
        num_batches = len(importances)
        num_layers = len(importances[0])
        batch_size = len(importances[0][0])

        # num_batches, num_layers, num_samples, 2 -> num_layers, num_samples*num_batches, 2
        layer_wise = [[] for _ in range(num_layers)]
        for batch_idx in range(num_batches):
            for layer_idx in range(num_layers):
                for sample_idx in range(batch_size):
                    layer_wise[layer_idx].append(
                        importances[batch_idx][layer_idx][sample_idx]
                    )

        # num_layers, num_samples, 2 -> num_samples, num_layers, 2
        sample_wise = [[] for _ in range(num_batches * batch_size)]
        for layer_idx in range(num_layers):
            for sample_idx in range(batch_size * num_batches):
                sample_wise[sample_idx].append(layer_wise[layer_idx][sample_idx])
        return sample_wise

    def get_random_samples_and_importances_across_all_layers(self, n_samples=1000):
        """Sample examples from validation dataset and find the corresponding importance scores
            across all layers.

        Args:
            n_samples (int, optional): The number of samples to be chosen. Defaults to 1000.

        Raises:
            ValueError: If  n_samples are greater than the samples in validation_dataset.

        Returns:
            tuple: The tuple containing the samples,
                word importance tuples and token importance tuples.
        """
        if n_samples > len(self.validation_dataset["input_ids"]):
            raise ValueError(
                "n_samples cannot be greater than the samples in validation_dataset"
            )
        np.random.seed(42)
        random_indices = list(
            np.random.choice(
                list(range(len(self.validation_dataset["input_ids"]))),
                size=n_samples,
                replace=False,
            )
        )
        self.train_validation_samples = self.train_datasets["validation"][
            random_indices
        ]
        samples = Dataset.from_dict(
            self.process_examples(self.validation_dataset[random_indices])
        )
        importances = self.get_importances_across_all_layers(samples)
        word_importances = self.rearrange_importances(importances["word_importances"])
        token_importances = self.rearrange_importances(importances["token_importances"])

        return samples, word_importances, token_importances

    def get_all_importances(
        self,
        load_from_cache=True,
        cache="/content/drive/My Drive/MLR/squad_ig_processed_dataset",
    ):
        """Get importance values for all samples across all layers.

        Args:
            load_from_cache (bool, optional): Whether to load processed examples
                from a cache if found. Defaults to True.
            cache (str, optional): The cache where processed examples are stored.
                Defaults to "/content/drive/My Drive/MLR/squad_ig_processed_dataset".

        Returns:
            tuple: The tuple containing the samples, word importance tuples
                and token importance tuples.
        """
        self.train_validation_samples = self.train_datasets["validation"]
        if load_from_cache and os.path.exists(cache):
            with open(cache, "rb") as in_file:
                samples = pkl.load(in_file)
        else:
            samples = Dataset.from_dict(self.process_examples(self.validation_dataset))
            with open(cache, "wb") as out_file:
                pkl.dump(samples, out_file)

        # columns = [
        #     "input_ids",
        #     "token_type_ids",
        #     "attention_mask",
        #     "start_positions",
        #     "end_positions",
        # ]
        # samples.set_format(type="torch", columns=columns, device="cuda",output_all_columns=True)

        # Cannot do the above because of GPU constraints

        importances = self.get_importances_across_all_layers(samples)
        word_importances = self.rearrange_importances(importances["word_importances"])
        token_importances = self.rearrange_importances(importances["token_importances"])

        return samples, word_importances, token_importances