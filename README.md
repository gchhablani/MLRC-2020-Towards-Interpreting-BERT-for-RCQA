# ML-Reproducibility-2020

This is our repository for the implementation of the paper [Towards Interpreting BERT for Reading Comprehension Based QA](https://openreview.net/forum?id=bpDFfs40geg&referrer=%5BML%20Reproducibility%20Challenge%202020%5D(%2Fgroup%3Fid%3DML_Reproducibility_Challenge%2F2020)) as a part of the [ML Reproducibility Challenge 2020](https://openreview.net/group?id=ML_Reproducibility_Challenge/2020).

## Usage


## Directory Structure


## Paper Summary

The paper talks about using Integrated Gradients (IG) to identify layer roles for BERT in Reading Comprehension QA tasks - SQuAD v1.1 and DuoRC SelfRC. They use the IG to create a probability distribution over the sequence for each example. Then they create Jensen-Shannon Divergence heatmaps across layers for 1000 samples keeping the top-2 tokens retained and top-2 tokens removed to see if layers focus on different words. Then, they use the token-wise imporances to create word-wise importances and use top-5 words to see what is the % of **predicted** answers, context words (within window size of 5 around answer) and query words in the passage in each layer. They observe that layers focus more on answer and contextual words and less on query words as the layers progress. This means that later layers focus on answer and words around the answer span, while initial layers focus on the query words and possible answers. Then, they plot an example based on the word-importances for layers and t-SNE representation for each layer's representation. Finally, they check the quantifier questions ('how much', 'how many') and observe that the ratio of numerical words in top-5 words increases as the layers progress. This is surprising as the confidence of BERT is still very high on such questions and the Exact Match scores are also high. A slightly more elaborate summary of the paper can be found here: https://gchhablani.github.io/papers/interpretBERTRC.
## Implementation

The paper uses original BERT script to train and evaluate the model on both the datasets, while we use custom scripts based on HuggingFace [datasets](https://huggingface.co/docs/datasets/) and [transformers](https://huggingface.co/transformers/) libraries.

Some salient differences between the two implementations:

**Differences with Original Bert SQuAD script**
| Original script                                                                                                                                                                                                                                                                      | Our Implementation                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Use a max query length of 64 for deciding the number of query tokens used.                                                                                                                                                                                                           | Doesn't consider max query length as we feel that the full question is needed. <span style="color:red">However, we will add a max query length option in the configuration soon.<span> |
| Their doc stride is based on a sliding window approach.                                                                                                                                                                                                                              | Our doc stride works on an overlap based approach, i.e. stride is the max overlap two features can have for an example.                                                                |
| Keep a track of the max_context features for the tokens using `score = min(num_left_context, num_right_context) + 0.01 * doc_span.length` so that they can filter start indices based on this.                                                                                       | We don't use max_context features, yet.                                                                                                                                                |
| Use a function to align the predictions after training. This function cleans the predicted answer of accents, tokenize on punctuations, and join the original text. Then the answer is stripped of spaces and compared and aligned with the original text in context and prediction. | We don't use any function  to clean the predictions after training which can significantly affect EM/F<sub>1</sub> scores.                                                             |

## Datasets and Training

We use HuggingFace [datasets](https://huggingface.co/docs/datasets/) and [transformers](https://huggingface.co/transformers/) libraries for training and evaluation of the models. We build our own dataset classes based using Dataset and DatasetDict classes internally.

We have added the datasets required - SQuAD v1.1 and DuoRC - by making submodules of their official repositories: [SQuAD](https://github.com/rajpurkar/SQuAD-explorer), [DuoRC](https://github.com/duorc/duorc) under the `data` directory, although we have only used the DuoRC files in our code. For SQuAD v1.1, we use [HuggingFace Dataset's squad](https://huggingface.co/datasets/squad) directly, but train [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) on it from scratch using the parameters as the [original BERT script](https://github.com/google-research/bert/blob/master/run_squad.py).

The DuoRC dataset has to be converted to SQuAD format before it can be use in any pre-trained model from HuggingFace.

On its own, DuoRC SelfRC has 30% questions without any answers in the context i.e. the answers are expected to be generated. SQuAD v1.1 on the other hand, has all the answers in the given passage, and the answer index and text is provided.

But for this paper, the DuoRC dataset has been converted to SQuAD format (with a start index and answer text). Our conversion isn't exactly same as the authors. The authors rely on Google Research's [original script](https://github.com/google-research/bert/blob/master/run_squad.py) for choosing the examples. We make choices while converting DuoRC to SQuAD format which are mentioned below.


### SQuAD
Processing SQuAD was relatively easier compared to DuoRC, as all the pre-trained models by HuggingFace are built on the SQuAD format. We directly use the dataset provided by HuggingFace, and use their tokenizers to return tokenized datasets for training, validation and prediction.

### DuoRC - Variant 1
We trained one of our models on SQuAD v1.1 format using this format. This gave a very high score, as many examples from the validation set were also dropped. <span style="color:red"> We discarded this model while performing the next steps.</span>

Example - 'Train S2' means that the dataset is Train and the format chosen for processing is SQuAD v2.0.
|                                     | Train S1.1 | Train S2   | Dev S1.1 | Dev S2   |
| ----------------------------------- | ---------- | ---------- | -------- | -------- |
| No Answer                           | Drop       | Keep       | Drop     | Keep     |
| Single Answer                       | Keep       | Keep       | Keep     | Keep     |
| Multiple Answers                    | Keep First | Keep First | Keep All | Keep All |
| Answer exists but not found in plot | Drop       | Keep       | Drop     | Keep     |

The model checkpoints/logs can be found here : <>

### DuoRC Modified - Variant 2
Here, we keep the no answers as empty in all training and validation sets, regardless of SQuAD v1.1 or SQuAD v2 style. We process the examples into SQuAD v1.1 format using the following logic. We have to do so in order to bring the F1 scores of the model closer to those reported in the paper, as well as the authors said that they didn't drop any examples in the validation set while prediction. Our analysis/results are based on this form of processing.

|                                     | Train S1.1 | Train S2   | Dev S1.1 | Dev S2   |
| ----------------------------------- | ---------- | ---------- | -------- | -------- |
| No Answer                           | Keep       | Keep       | Keep     | Keep     |
| Single Answer                       | Keep       | Keep       | Keep     | Keep     |
| Multiple Answers                    | Keep First | Keep First | Keep All | Keep All |
| Answer exists but not found in plot | Drop       | Keep       | Keep     | Keep     |

## Analysis and Results


### Integrated Gradients

The authors use a custom implementation of Integrated Gradients, with `m_steps = 50` and over all the examples in SQuAD and DuoRC. Due to differences in our initial understanding of the paper, we have implemented Integrated Gradients (IG) using the PyTorch-based library - [Captum](https://captum.ai/docs/extension/integrated_gradients).

The authors calculate Integrated Gradients on each layer's input states and use reimann-right numerical approximation. They calculation attributions of the layers on maximum of softmax of start and end logits separately with m_steps = 50. Due to computational restrictions, we had to reduce the number of samples and steps we could calculate Integrated Gradients on.

Additionally, they consider the best feature for each example predicted by the model only for finding out the importance values.

**Differences w Authors' Integrated Gradients**
| Authors' Implementation                                               | Our Implementation                                           |
| --------------------------------------------------------------------- | ------------------------------------------------------------ |
| m_steps = 50                                                          | m_steps = 25                                                 |
| n_samples = size of the validation dataset                            | n_samples = 1000                                             |
| one feature per example (best predicted)                              | can have multiple features per example.                      |
| target is max(softmax(start_logits)) for start and similarly for end. | target is max(start_logits) for start and similarly for end. |

They norm the token attributions generated and normalize it to get a probability distribution. They use this probability distribution to calculate word-wise importances by adding importances for tokens of each word together, and re-normalizing the scores.

<span style="color:red;">Since we take multiple features per sample, we can't directly use the "predicted answer" as the answer word for each feature. Hence, we predict again on the chosen samples and take feature-wise (instead of example-wise) best processed predictions for each feature as the answer.</span>

<span style="color:red;">Due to differences in the method of IG, our word-categories for word importances are different, as well as token importances which can bring some differences to the statistics calculated from the paper.</span>

We plan to make these changes to our repository soon in order to bring the code as close to authors' implementation of the approach.

Note: In other variants, targets can be either: 

- argmax(softmax(logits)) for start and end
  
- argmax(logits) for start and end
  
- best start and end logits based on max(start_logits+end_logits)  
  
- best start and end logits based on max(softmax(start_logits)+softmax(end_logits))
  
- ground truth start and end

### Jensen Shannon Divergence
The authors calculate Jensen-Shannon Divergence using the [dit](https://dit.readthedocs.io/en/latest/measures/divergences/jensen_shannon_divergence.html) library. For 1000 examples, they retain top-2 token importances and zero out the rest and plot heatmap for inter-layer divergence values.
They remove top-2 token importances and keep the rest and again plot the heatmap. They observe that the heatmap with top-k retained importances has higher gap in max and min, meaning layers focus on different words while for the top-k removed case, they see almost a uniform distribution.

We repeated this with 1000 features, instead of examples, and observe similar heatmaps.



The essence of this analysis is to look at the gap between max and min values in the heatmap, and which pair of layers have similar top-k importances, and which pair of layers have different top-k importances.
### QA Functionality


### Qualitative Examples


### Quantifier Questions


