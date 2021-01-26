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


## QnA with the authors

1. BertTokenizer usually breaks a word into multiple tokens due to WordPiece Embeddings. In that case, for some words there will be multiple vectors for each layer. A simple way to combine these would be to average them for a word. How was this handled in the implementation?

   - We keep the embeddings for the different segments of a word separate, and calculate separate integrated gradient scores for them, which we then normalize to get importance scores. Later, we add up the importance scores of all these segments to get the overall importance score of the word. [The segments can be identified by searching for a "##" symbol in the word - this can be checked and confirmed by printing out the passage words]. 

2. Bert Base has a max sequence length of 512 tokens. For DuoRC SelfRC the max length of tokens for train is 3467 per passage, with the mean of 660. Similarly, for SQuAD v1.1 the max length is 853, with the mean of 152. For each of these, is the max length set to 512? If that is done, then is only the article/passage/context truncated? If yes, how?

    - We maintain the max length of 384 tokens in both SQuAD and DuoRC in our experiments. 

3. For Duo RC, there are cases where there are multiple answers to a question, and good number of cases where the answer is not present in the passage, what is done regarding these cases during the implementation?
Example for Multiple Answers:

    ```['Maurido', 'Mauricio']
    ['to own a hotel', 'to own his own hotel']
    ['Tijuana, Mexico', 'Tiajuana']
    ['Tessa.', 'Tessa', 'Tessa']
    ```

   - We use the available tensorflow implementation of BERT, which handles multiple answers by itself. Multiple answers are seen in SQuAD as well as DuoRC.

4. How did you find numerical words/tokens in the passage/question and quantifier questions? I checked a library called word2number but it only works for number spans, and only when it is an exact number word. I couldn't find any popular approaches.

   - We use NLTK POS tagging on the passage words, and the words which have the POS tag of 'CD' (cardinal) are taken to be the quantifier words. On the question side, we take questions which have the words "how many" or "how much" as quantifier questions.

5. What is the base used for Jensen-Shannon Divergence? The units or log base.

   - We use the implementation of jensen_shannon_divergence from the library dit.divergences . Please check the documentation, I am unable to recollect the exact details now.
"from dit.divergences import jensen_shannon_divergence"

6. How was the contextual passage for t-SNE visualization decided? Was this supposed to be the whole sentence that contains the answer "span"?

   - We chose words within a distance of 5 words on either side of the answer span as contextual words for tables. The whole sentence was chosen for t-SNE visualization.

7. What were the other training/fine-tuning choices made, with respect to hyperparameters, optimizers, schedulers, etc.?

   - We used the default config given in BERT's official code. However, we changed the batch size to 6 to fit our GPU.

8. What is EM in `87.35% EM`? (mentioned in Section 5.2 in the Quantifier Questions subsection)

   - To measure the performance of the model on answer spans, both SQuAD and DuoRC use the same code - with 2 metrics : F1 score and Exact Match (EM). The 87.35% EM refers to the exact match score.

9. The paper mentions that all answer spans are in the passage. While that is true for SQuAD, DuoRC has answers not present in the passage. Did you remove such cases?

    - Yes, we remove train cases where the answer is not in the passage (this is done by the BERT code itself). However, we do not remove any data points from the dev set.

10. I have another doubt regarding t-SNE representations. For multi-token words, do you take the average of all those token representations as the word representation while plotting?

    - tSNE was a qualitative analysis, and for the examples we picked, we didn't observe segmentation of words. If you'reanalyzing examples with segmentation, I guess you could try both merging the embeddings, or keeping the different segments separate.

11. When calculating Integrated Gradients, for start and end there will be different attribution values for each word representation (because we have two outputs for each input), how was it combined when calculating the IG for each word?

    - We calculate the IG of both the start probabilities and the end probabilities with respect to the word embedding at hand, and then add them up.

12. I store tokens, token_wise_importances, words, and word_wise_importances (after removing special tokens and combining at ##)
The JSD was built on token wise distributions or word wise distributions?

    - JSD was on token wise with length 384, table on word wise importances

13. Should the IG be calculated on ground targets or predicted outcomes?

    - We calculated the attributions of what the model *has* predicted, rather than what it *should have* predicted. We followed another attention analysis based paper for this logic.

14. What if a particular word is a query word and also in the contextual span (within window size 5 of the answer)?

    - I just consider them twice then.. if a word was both a query word and a contextual word, it probably would have served dual functionality in the training as well, I guess.
    - While finding query words, remove the stopwords from the search.

15. Should the stopwords be removed for query as well as contextual words? Should the window size be applied after removing stopwords or before? Should the top-5 words contain stopwords?

    - Keep them for contextual. Because they are actually part of the context. But when finding question words in the passage, ignore the stop words in the question because you'll probably find many "is" or "the" or etc in the passage and all they needn't correspond to the query.

    - Should top-5 words include stopwords? - here it's okay

16. The answer spans/answers in the analysis are actual answers right? 

    - again, we chose the answer which the model predicted, not the actual answer span (same logic as used for IG).
  
17. Did you take predicted answers for tables and t-SNE as well?
    - I used predicted (and processed) answers for all the analysis after training.
