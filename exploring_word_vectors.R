## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'19 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi19/lasi19-workshops-tutorials/)


## The script is aimed at familiarizing with word vectors (aka word embeddings) and 
## some of their key features. These features are exemplified on a pre-built word 
## embeddings model, based on the GloVe method for learning word embeddings.
##
## GloVe stands for Global Vectors for Word Representation.
## It is an unsupervised learning method for obtaining vector representation of words.
## It was introduced in the paper:
## Pennington, J., Socher, R. & Manning, C. D. (2014). Glove: Global Vectors for Word 
## Representation. EMNLP.(pp.1532-1543). URL: http://nlp.stanford.edu/pubs/glove.pdf
## 
## Pre-trained GloVe models are available for download from:
## https://nlp.stanford.edu/projects/glove/ 
## We will use one of the models bundled within the "glove.6B.zip" data file. 
## Some facts about the glove.6B models:
## - they were trained on the corpus that resulted from combining a 2014 Wikipedia dump
##   and Gigaword5, and consisted of 6 billion tokens (hence 6B in the name)
## - after tokenizing and lowercasing the corpus, 400K most frequent words were used
##   to build the vocabulary
## - the zip includes models with 50, 100, 200, and 300 dimension vectors 
##   (each one in a separate .txt file) 


###################
##
## SETUP a SESSION
##
###################

# For the exploration of word vectors, we will need the following R packages:
# - wordVectors
# - dplyr
# - ggplot2, ggrepel
# - tsne, Rtsne 

# If you miss any of these packages, install them, before proceeding with the script
# install.packages(c("<package_name_1>", "<package_name_2>", ...))
# NB: the wordVectors package is not available from CRAN, so, it cannot be installed 
# in the usual way, but have to be installed using devtools. For details, please see
# the installation instructions at: https://github.com/bmschmidt/wordVectors

# About the required R packages:
# - wordVectors: an R package for building and exploring word embedding models
# - ggrepel: an addition to ggplot; provides text and label geoms for 'ggplot2' that help to avoid 
#   overlapping text labels; labels repel away from each other and away from the data points.
# - tsne - a "pure R" implementation of the t-SNE (T-distributed Stochastic Neighbor Embedding)
#   algorithm for dimensionality reduction
# - Rtsne - an R wrapper around the fast t-SNE implementation by Van der Maaten (the creator of t-SNE)

# We will initially load only the package that will be used throughout this script
# and include additional ones as they become needed
library(dplyr)

# Also, load a script with some auxiliary functions
source("word_vec_utils.R")

##############################################
##
## LOAD a PRE-TRAINED GLOVE WORD VECTOR MODEL 
##
##############################################

# Load a pre-trained GloVe word vectors model. In particular, we'll use the glove.6B.300d 
# model - the model with 300 dimension word vectors. 
# NB. Regarding the vector dimensions, higher vector dimension is often associated with 
# higher precision, but also tends to include more random error and slower operations. 
# Likely choices are in the range 100-300.

# Read in the the model 
# (Note: change the 'glove_6B_300d_dir' variable to the path of the directory 
#  where the "glove.6B.300d.txt" file is stored on your computer)
# glove_6B_300d_dir <- "~/R Studio Projects/Large datasets/glove.6B/"
glove_6B_300d_dir <- "C:\\Users\\jovanje\\Documents\\LASI 2019\\glove.6B\\"
g6b_300d <- scan(file = paste0(glove_6B_300d_dir, 'glove.6B.300d.txt'), what="", sep="\n")

# What we have read - g6b_300d - is in fact a huge character vector, 
# consisting of 400K entries - one entry per word from the vocabulary. 
g6b_300d[1]
# Each entry is given as a string that consists of 301 items
# delimited by a space: the 1st item is a word and the rest (300 items)
# are the estimated values of the 300 dimensions of that word

# Create a data frame out of the large vector read from the file
# (get_word_vectors_df() is defined in the word_vec_utils.R script)
g6b_300d_df <- get_word_vectors_df(g6b_300d, verbose = TRUE)
dim(g6b_300d_df)
View(g6b_300d_df[1:20, 100:120])

# Remove g6b_300d to release memory
remove(g6b_300d)

###########################
##
## EXPLORE the GLOVE MODEL 
##
###########################

library(wordVectors)
library(tsne)

# Why would we want to explore a word embeddings model?
#
# a) To learn something about the *sources* (corpus used for learning word vectors) by 
#    examining the use of language. 
# b) To detect potential biases built into the model that we intend to use for a text mining task; 
#    this can be both technically and ethically important.
# The exploration step is particularly important if you are not sufficiently familiar with 
# the text corpus that was used for learning the embeddings model (ie. word vectors).


# In order to explore the loaded GloVe model using some handy functions 
# from the wordVectors package, we need to transform the g6b_300d_df
# data frame into a wordVectors::VectorSpaceModel object.
# The first step is to transform the data frame to a matrix with 
# - words in the rows and vector dimensions in the columns
# - row and column names defined as follows: row names should be words whose   
#   vectors are stored in the corresponding rows; column names are typically
#   set to "V1", "V2", ..., "Vn", where n is the vector dimension
g6b_300d_matrix <- t(as.matrix(g6b_300d_df)) 
dim(g6b_300d_matrix)
rownames(g6b_300d_matrix) <- colnames(g6b_300d_df)
colnames(g6b_300d_matrix) <- paste0("V", 1:300)
# Next, create an object of the VectorSpaceModel class 
glove_model <- new("VectorSpaceModel", g6b_300d_matrix)
# Since we won't need the matrix and df representation of the Glove model,
# we can remove them to release memory
remove(g6b_300d_matrix, g6b_300d_df)

# Start exploring the vector space by examining the "neighbourhood" of a
# chosen word, that is, words that are closest to the given word (in the
# multi-dimensional vector space):
?closest_to
glove_model %>% closest_to("canada", n=20, fancy_names = FALSE)
glove_model %>% closest_to("awesome", n=20, fancy_names = FALSE)

# A note regarding the closest_to function and the notion of 'closeness'
# in a vector space:
# the computed similarity scores are cosine similarities in the vector 
# space: 1.0 means perfect similarity, 0 is no correlation, and -1.0 is 
# complete opposition. However, "opposition" in a word vector space is 
# often different from the colloquial use of "opposite".
# For example, 10th closest word to "awesome" is "awful"; the reason:
# those two words fullfil the same role - they are used to describe 
# some experience, opinion, taste, and the like; this kind of 'role-based
# similarity' can be observed in the above given 20 closest words to 
# "awesome".


# We can use t-SNE to reduce the dimensionality of the word vector model 
# to 2, so that we can visualize it in 2D space. 
# t-SNE stands for T-distributed Stochastic Neighbor Embedding
# (https://lvdmaaten.github.io/tsne/). It is a recently discovered 
# dimensionality reduction technique (like PCA, but better); it is
# particularly suitable for the visualization of high-dimensional datasets. 
# If interested in getting an intuition regarding how t-SNE works,
# you may find this YouTube video helpful: 
# https://www.youtube.com/watch?v=NEaUSP4YerM


# Take 500 words closest to the word "canada":
canada_closest_500 <- glove_model %>% 
  closest_to("canada", n=500, fancy_names = FALSE) %>%
  pull(word)
# Create and plot a subset of the overall model, which includes only 
# the "canada"'s 500 closest neighbours 
glove_model[[canada_closest_500, average = FALSE]] %>% 
  plot(method="tsne")

# a note regarding the 'average' parameter: if set to TRUE, the result will be 
# one vector obtained by averaging vectors corresponding to the selected words
# (in this case, "canada"'s 500 closest neighbours)

# This visualization is not easy to read and undrestand due to frequent 
# word overlaps. So, we'll use another function that combines t-SNE with 
# the ggplot2 and ggrepel packages for better plotting 
# (the function is defined in the word_vec_utils.R script):
w2v_plot(model = glove_model, 
         search_terms = "canada", 
         n_nearest = 500, 
         out_dir = "results")
# (note: the function stores the visualization as a jpeg file in the out_dir folder
# and it will expect for such a folder to exist)


# We can also look for words that are closest to a combination of some other words.
# For example, we can look for words that are closest to "canada" and "awesome":
glove_model %>% closest_to(~ 'canada' + 'awesome', n=20)

# We can also look for words that are close to 'canada' but not to 'awesome' by 
# using subtraction:
glove_model %>% closest_to(~ 'canada' - 'awesome', n=20)


# Let's also take a quick look at the analogy task that word vectors are famous for

# An often cited example is the one with countries and their capitals:
# vector('Paris') - vector('France') + vector('Germany'):
glove_model %>% closest_to(~ "paris" - "france" + "germany")

# How to interpret these arithmetics?
# One interpretation may be that we are starting with "paris", removing
# its similarity to "france", and adding a similarity to "germany".

# Another way to consider the same vector arithmetics is as follows:
glove_model %>% closest_to(~ "paris" + ("germany" - "france"))
# You have the vector `("germany" - "france")` that stands for things that are 
# associated with 'germany' but not with 'france'; we then add this vector to "paris", 
# and that way we move to a new neighborhood of things that are common to 'paris' 
# and 'germany' (stripped of 'france')

# We can also use this kind of arithemtics to move along a gender vector:
glove_model %>% closest_to(~ "brother" + ("she" - "he"))


# For additional insights into wordVectors' functions for exploring word vector
# models, examine the wordVectors' Exploration vignette that demonstrates 
# those functions on a model trained on teaching evaluations:
# https://github.com/bmschmidt/wordVectors/blob/master/vignettes/exploration.Rmd

#################################################
##
## OTHER WELL-KNOWN PRE-BUILT WORD VECTOR MODELS
##
#################################################

#
# Word2Vec
#
# Prebuilt models (in English) available from:
# https://code.google.com/archive/p/word2vec/ 
# 
# Can be loaded into R with the read.vectors() function 
# from the wordVectors package.


#
# FastText
# 
# Prebuilt models (in 157 languages) available from:
# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md#models
#
# Can be loaded into R using base R functions; for instructions see:
# https://stackoverflow.com/questions/50569420/use-a-pre-trained-model-with-text2vec


#
# A comprehensive reference of various kinds of word vector models (*2vec) 
# is available at:
# https://gist.github.com/nzw0301/333afc00bd508501268fa7bf40cafe4e
# 