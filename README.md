# LASI'19 workshop on Text Mining for Learning Content Analysis

This repository stores materials for the **Text mining for learning content analysis** workshop organized at the [Learning Analytics Summer Institute 2019 (LASI'19)](https://solaresearch.org/events/lasi/lasi19/lasi19-workshops-tutorials/), held at University of British Columbia, Vancouver, Canada, on June 17-19, 2019.

The stored R scripts cover 3 topics: 

1. General text mining (TM) workflow exemplified through a binary text classification task. It covers the overall TM process, starting with text preprocessing and ending with examination and evaluation of the results. Scripts covering this topic:
..* preprocess_20News_dataset.R
..* newsgroup_classifier.R
..* tm_utils.R

2. Introduction to word vectors (word embeddings). The aim is to familiarize with the notion of word vectors through exploration of a pre-built word vector model. In particular, [GloVe model](https://nlp.stanford.edu/projects/glove/) (w/ 300 dimensions) is used. The same pre-built model is used for the classification task in topic 3. [T-sne](https://lvdmaaten.github.io/tsne/) dimensionality reduction technique is used for visualization of word vectors in 2D space. Relevant scripts are:
..* exploring_word_vectors.R
..* word_vec_utils.R

3. Using word vectors for text classification. Two ways of using a pre-built word vector model for document classification are exemplified: i) using weighted average of word vectors to form document vectors; ii) using [Word Mover Distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) to compute the similarity of documents based on their word vectors. Scripts that cover this topic:
..* newsgroup_GloVe_classifier.R
..* tm_utils.R
..* word_vec_utils.R

Note also that some prebuilt models are available in the 'models' folder. They are made available so that we do not need to wait for models to build during the workshop.

The first and third topic are based on the [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/). This dataset, widely used in text mining tasks and benchmakrs, is a collection of approximately 20,000 newsgroup documents (forum posts), partitioned (nearly) evenly across 20 different newsgroups, each corresponding to a different topic. The csv files, in the data/20news folder, are derived from this dataset (subsetted and pre-processed).

Slides that introduce relevant concepts and methods are available at the links given below. The slides also cover some recent research work in Learning Analytics that was either partially or fully based on TM methods and techniques.  
* [Text Mining (workflow) for Learning Content Analysis](https://1drv.ms/p/s!AjwXFgNk6IQbhEWvmfr1R-0uYvJx)
* [Bagging and Random Forest](https://1drv.ms/b/s!AjwXFgNk6IQbgh5G-vQCyWnaXwZL)

If interested in learning more, you may want to check materials from the [previous edition of this workshop](https://github.com/jeljov/Text_Mining_at_LASI18), held at LASI'18.