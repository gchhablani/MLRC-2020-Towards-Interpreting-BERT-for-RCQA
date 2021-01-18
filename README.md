# ML-Reproducibility-2020

This is our repository for the implementation of the paper [Towards Interpreting BERT for Reading Comprehension Based QA](https://openreview.net/forum?id=bpDFfs40geg&referrer=%5BML%20Reproducibility%20Challenge%202020%5D(%2Fgroup%3Fid%3DML_Reproducibility_Challenge%2F2020)) as a part of the [ML Reproducibility Challenge 2020](https://openreview.net/group?id=ML_Reproducibility_Challenge/2020).

## Datasets

We have added the datasets required - SQuAD v1.1 and DuoRC - by making submodules of their official repositories: [SQuAD](https://github.com/rajpurkar/SQuAD-explorer), [DuoRC](https://github.com/duorc/duorc) under the `data` directory.

Additionally, the official evaluation script for SQuAD has been added to `scripts` using the following terminal command:

```sh
wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O ./scripts/evaluate.py
```


**DuoRC Processing**
We process the examples into SQuAD format using the following logic. 
Example - 'Train S2' means that the dataset is Train and the format chosen for processing is SQuAD v2.0.
|                                     | Train S1.1    | Train S2      | Dev S1.1      | Dev S2        |
| ----------------------------------- | ------------- | ------------- | ------------- | ------------- |
| No Answer                           | Keep as empty | Keep as empty | Keep as empty | Keep as empty |
| Single Answer                       | Keep          | Keep          | Keep          | Keep          |
| Multiple Answers                    | Keep First    | Keep First    | Keep All      | Keep All      |
| Answer exists but not found in plot | Drop          | Keep          | Keep          | Keep          |