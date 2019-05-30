## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'19 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi19/lasi19-workshops-tutorials/)


## The script provides an example of the overall process of text classification, 
## including: 
## - preprocessing of textual data;
## - transformation of unstructured (textual) data into a structured data format that
##   can be fed into a classification algorithm; this includes feature weighting and 
##   selection, as well as methods for reducing / transforming the feature space,
##   that is, turning a large number of sparse features into a significantly smaller 
##   number of dense features;
## - application of classification algorithms on the transformed textual data 
##   (that is, the created feature set);
## - evaluation of the classification results.
##
## The example is based on a subset of the 20 Newsgroups dataset.

## Even though the 20 Newsgroup dataset allows for multiclass classification,
## to make this example easier to follow and understand, we will limit ourselves 
## to a binary classification task. Note that the same procedure is applicable  
## to multiclass classification, only the computation process would be more 
## demanding and thus the computation time longer. 

## For the binary classification task, we will use the group discussing guns 
## ('talk.politics.guns') and the one on Mideast ('talk.politics.mideast'), both within 
## the broader 'politics' topic. The rationale for choosing these two groups:
## - The groups' discussions cover the topics that can be expected in a discussion forum
##   of a course dealing with contemporary political issues.  
## - Being topically closely related, the two groups will pose a challenge for a 
##   classifier - it is not an easy task to differentiate between groups of posts 
##   where topics are closely related and a shared vocabulary is to be expected 
##   (at least far more than in the case of differentiating between posts on, 
##   e.g., space and medicine).     


##############################################
## SET UP the SESSION: 
## - INSTALL and LOAD the REQUIRED LIBRARIES
## - LOAD the AUXILIARY SCRIPTS
##############################################

# The script makes use of the following R packages:
# - caret, e1071 - for various ML tasks
# - rpart - for building a decision tree classifier
# - ranger - for building a Random Forest classifier
# - pROC - for computing ROC-related evaluation measures 
# - quanteda - for various text analytics tasks
# - stringr - for advanced string processing
# - irlba - for singular vector decomposition (SVD)
# - dplyr, tidyr - for general data analysis tasks
# - ggplot2 - for visualization
# - janitor - for some data cleaning tasks
# - doSNOW - for multi-core parallel process execution 
# If you miss any of these packages, install them, before proceeding with the script
# install.packages(c("<package_name_1>", "<package_name_2>", ...))

# Initially, we will load just a basic set of R packages 
# whereas the others will be loaded along the way, as we need them
library(dplyr)
library(readr)
library(tidyr)

# Load a set of auxiliary functions
source("tm_utils.R")

# Set the seed to be used in various computations that depend on random processes.
# This is to assure reproducibility of the results.
seed <- 6219


##########################################################
## LOAD CLEANED DATA from csv files and
## SELECT TWO NEWSGROUPS for a BINARY CLASSIFICATION TASK
##########################################################

# We’ll start by reading in the 'cleaned' newsgroup dataset, that is, by 
# reading the CSV files that resulted from preprocessing the original dataset
# through the steps implemented in the preprocess_20News_dataset.R script. 
# The 'cleaned' data were obtained by removing some 'extra' text that 
# we will not need for this analysis, including: 
# - post header
# - automated email signatures 
# - lines of text referencing quotations from other users
# NB.1: since this 'extra' text has been removed using some simple heuristics,
# it might happen that not all these extra text bits were removed. 
# NB.2: data stored in the header of a post may be useful for
# classification purposes. However, in this case, we will restrict
# our analysis to the content of the discussion.

# Load training data
train_posts <- read_csv("data/20news/20news-bydate-train.csv") %>% 
  data.frame()
glimpse(train_posts)
View(head(train_posts, 10))

# Load test data
test_posts <- read_csv("data/20news/20news-bydate-test.csv") %>%
  data.frame()
glimpse(test_posts)
View(head(test_posts, 10))

# Keep only the data related to the two chosen newsgroups
selected_ngs <- c('talk.politics.guns', 'talk.politics.mideast') 
selected_lbls <- c('guns', 'mideast')

train_2cl <- train_posts %>%
  filter(newsgroup %in% selected_ngs) %>%
  mutate(newsgroup = factor(newsgroup, # transform newsgroup into a factor w/ shorter labels
                            levels = selected_ngs,
                            labels = selected_lbls))
test_2cl <- test_posts %>%
  filter(newsgroup %in% selected_ngs) %>%
  mutate(newsgroup = factor(newsgroup, 
                            levels = selected_ngs,
                            labels = selected_lbls))


# We will now use the training set to build a classifier.
# Test set will be used later, only for evaluation purposes.

####################################
## DATA (TEXT) WRANGLING: 
## TEXT CLEANING AND TRANSFORMATION 
####################################

# There are many packages in the R ecosystem for performing text analytics.
# One of the latest is *quanteda*. It has many useful functions for quickly
# and easily working with text data; they are well explained in the
# quanteda docs: https://quanteda.io/
library(quanteda)

#
# Tokenization of posts
#

# When tokenizing documents, a typical practice is to remove punctuation marks
# and symbols; also, numbers and urls are often removed. 
# Here, we'll keep numbers and urls and deal with them later. 
?tokens
train_tokens <- tokens(x = train_2cl$post_txt, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE)

# Take a look at a specific post and see how it has been transformed
train_2cl$post_txt[11]
train_tokens[[11]]

# Note that by removing punctuation we 'lost' date in the example post.
# If we would want to preserve dates and similar kinds of information
# (ie. extract them as separate features), we would need to do that
# (eg. using regular expressions) before removing punctuation.

# Next, we will reduce all tokens to lower letters to reduce the variability of 
# the token set (a part of the process known as text normalization)
train_tokens <- tokens_tolower(train_tokens)
train_tokens[[11]]

# Replace numbers (if present) with the "NUMBER" entity.
# The rationale: we are not interestd in specific numbers, just the fact that a
# number appeared in a post
train_tokens <- tokens_replace(x = train_tokens, 
                               pattern = "^[0-9]+(-[0-9]+)?$", 
                               replacement = "NUMBER",
                               valuetype = 'regex',
                               verbose = TRUE)
train_tokens[[11]]


# Next, replace urls (if present) with the "URL" entity
# The same kind of rationale as for numbers.
# First check if they are present
url_pattern <- "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
url_tokens <- tokens_select(x = train_tokens,
                            pattern = url_pattern, 
                            valuetype = "regex",
                            verbose = TRUE) 
# no URLs...
# NB: the used URL regex pattern is taken from: http://urlregex.com/
# It is not a perfect one, but covers majority of cases.


# Inspection of a sample of posts indicated that email addresses are often present among 
# the extracted tokens; we will remove them, though this might not be necessary as they 
# will probably be removed later (after applying a word weighting scheme) due to their 
# low relevance/weight 
email_pattern <- "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
train_tokens <- tokens_remove(x = train_tokens, 
                              pattern = email_pattern, 
                              valuetype = "regex", 
                              verbose = TRUE)

# NB. the applied regex pattern is taken from: https://emailregex.com/ ;
# an alternative one: "^[\\w-\\.]+@(\\w+[\\.-]?)+\\w{2,4}$"

train_2cl$post_txt[9]
train_tokens[[9]]

## Note: regular expressions are very handy and often indispensable for text cleaning and
## transformation. If you feel you need to learn about regex or refresh your memory, 
## this tutorial is excellent: http://regex.bastardsbook.com/
## Also, the following R cheatsheet comes in useful:
## https://www.rstudio.com/wp-content/uploads/2016/09/RegExCheatsheet.pdf 


# Looking again at the example post, one may observe the presence of tokens with 1 or 2 
# characters only; these will be removed as they rarely bear any meaning
train_tokens <- tokens_keep(x = train_tokens, min_nchar = 3)
train_tokens[[11]]

# Note: you may also consider removing overly long tokens, but as these would be
# rather rare (ie. have very low frequency), they would be, anyway, removed in the
# feature selection/filtering phase.


# Since forum posts, as well as messages exchanged in other kinds of online  
# communication channels (e.g. chat, status posts), tend to have misspelled 
# words, it might be useful to do spelling correction, as a part of the text 
# normalization step. A typical approach is to check the text against some of 
# the available misspelling corpora 
# (e.g. http://www.dcs.bbk.ac.uk/~ROGER/corpora.html).
# There is also an R package - spelling - for spell checking:
# https://cran.r-project.org/web/packages/spelling/index.html
# We will skip this step for now.


# Next, we will remove stopwords.
# To that end, we will use quanteda's default stopwords list for English.
?stopwords
# It is advised to inspect the default stopword list before
# applying it to the problem at hand - the default one may not be suitable
# for the task at hand (e.g. for sentiment / emotion detection personal  
# pronouns are typically preserved).
head(stopwords(), n = 50)
tail(stopwords(), n = 50)
train_tokens <- tokens_remove(train_tokens, stopwords())
train_tokens[[11]]

## A few notes: 
## -  quanteda uses stopwords R package as the source for stopwords lists:
##    https://github.com/quanteda/stopwords
## -  depending on the task at hand, you might want to extend the built-in
##    stopword list with additional, corpus specific 'stopwords'
##    (e.g. overly frequent words in the given corpus).


# Perform stemming on the tokens
# (the function uses Porter's stemming algorithm) 
train_tokens <- tokens_wordstem(train_tokens, language = "english")
train_tokens[[11]]


# In case you need lemmatisation for the task at hand,
# consider using the *udpipe* R package:
# https://github.com/bnosac/udpipe
# as it offers language models for a number of languages 
# and using such models, you can lemmatise words

###################################
# CREATE DOCUMENT TERM MATRIX (DTM)
###################################

# Now, we are ready to create DTM. 
# In quanteda's terminology DTM is referred to as "document feature matrix" or dfm
?dfm
train_dfm <- dfm(x = train_tokens, 
                 tolower = FALSE)  # we've already lower cased the tokens

train_dfm
# It's very sparse (sparsity = the proportion of cells that have zero counts); 
# we can get the precise level of sparsity with:
sparsity(train_dfm)

# By default, words are weighted with term frequencies (TF)
View(as.matrix(train_dfm)[1:20,1:20])

## However, TF is not always the best metric for estimating the relevance of a
## word in a corpus. The Term Frequency-Inverse Document Frequency (TF-IDF) metric 
## tends to provide better results. Specifically, TF-IDF accomplishes the 
## following goals:
## - The TF metric does not account for the fact that longer documents will have 
##   higher individual term counts. By normalizing TF values, using, for example, 
##   L1 norm, that is, the document length expressed as the number of words,
##   we get a metric that is length independent.
## - The IDF metric accounts for the frequency of term appearance in all documents 
##   in the corpus. The intuition being that a term that appears in almost every  
##   document has practically no predictive power.
## - The multiplication of (normalized) TF by IDF allows for weighting each term 
##   based on both its specificity at the level of the overall corpus (IDF) and its 
##   specificity for a particular document (i.e. relatively high presence in the document).

# Hence, we'll compute TF-IDF values for the terms as estimates 
# for the terms' relevance
train_dfm <- dfm_tfidf(train_dfm, 
                       scheme_tf = 'prop', # L1 normalization
                       scheme_df = 'inverse') # default option

# Given the large number of features (~16.6K) and the high level of
# sparsity, we should consider doing feature selection. 

# Examine total (corpus-level) TF-IDF value for each word
summary(colSums(train_dfm))
# Summary stats suggest very uneven distribution; 
# let's inspect it visually, to better understand its extent 
# (note: the plotting function is defined in tm_utils.R)
plot_word_weight_distr(wweights = colSums(train_dfm), 
                       lbl = "TF-IDF distribution for unigrams",
                       bin_width = 0.05)
# Considering the large number of features and their low total weights, 
# we will keep features with total TF-IDF above the 75th percentile
tfidf_total <- colSums(train_dfm)
to_keep <- which(tfidf_total > quantile(tfidf_total, probs = 0.75))
train_dfm_reduced <- dfm_keep(train_dfm, 
                              pattern = names(to_keep),
                              valuetype = "fixed", 
                              verbose = TRUE)
train_dfm_reduced
# a significant reduction: from ~16.6K to ~4K features

# Let's also examine the number of documents each token appears in 
# (i.e., document frequency - DF), to check if there might be tokens 
# that are present in a large proportion of documents 
# (e.g. in over 60% of documents)
dfreq <- apply(train_dfm_reduced, 2, function(x) sum(x>0))
summary(dfreq)
sort(dfreq, decreasing = TRUE)[1:20]
# The NUMBER feature, with its presence in near 60% of posts, might
# prove irrelevant (too common), but let's keep it for now

# Next, we use the (reduced) DTM to setup a feature data frame 
# with (class) labels. It will serve as the input for a 
# classification algorithm. 
# To create such data frame, we need to do the following:
# - transform quanteda's dfm to a 'regular' R data.frame
# - check for (and remove) documents (rows) that have 'lost' 
#   all their words in the feature selection step
# - assure that feature (column) names are regular R names
# This is done by the create_feature_df() f., defined in the 
# tm_utils.R script
train_df <- create_feature_df(train_dfm = train_dfm_reduced, 
                              class_labels = train_2cl$newsgroup)


############################################################
# BUILD the 1st ML MODEL: RPART + UNIGRAMS + TF-IDF WEIGTHS
############################################################

# As per best practices, we will leverage cross validation (CV) for our
# modeling process. In particular, we will perform 5-fold CV to
# tune parameters and find the best performing model.
# (NB. we restrict ourselves here to 5-fold CV so that the training 
# does not last overly long. When working on your own, better use 
# higher number of folds, typically 10 folds, and even do repeated CV
# to get a better estimation of the model performance).

# The *caret* package will be used for model building through CV
library(caret)

# Note that our data set is not trivial in size. As such, depending on the
# chosen ML algorithm, CV runs might take a long time to complete. 
# To cut down on the total execution time, we use the *doSNOW* R package 
# to allow for CV to run in parallel on multiple (logical) cores.
# Parallel processing, for model building and CV, is directly supported 
# in caret: https://topepo.github.io/caret/parallel-processing.html

# Due to the size of the DTM, at this point, we will use a single decision
# tree (DT) algorithm to build our first model. We will use more powerful algorithms 
# later when we perform feature reduction to shrink the size of our feature set.

# Load also the *rpart* R package required for building DTs 
library(rpart)

# We will tune the cp parameter, which is considered the most important in the
# rpart function (the function used in the rpart package for building a DT).
# cp stands for the complexity parameter; any split that does not improve the overall
# fit of the model by at least cp is not attempted; default value is 0.01.

# Define the grid of values for the cp parameter to be examined during the CV process
cp_Grid = expand.grid( .cp = seq(from = 0.001, to = 0.02, by = 0.001)) 

# Build a DT classifier through CV 
# (the cross_validate_classifier() function is defined in the tm_utils.R script)
rpart_cv_1 <- cross_validate_classifier(seed,
                                        nclust = 5, # 7 would be the best value, according to caret's doc.
                                                    # however, not possible on my laptop
                                        train_data = train_df,
                                        ml_method = "rpart",
                                        grid_spec = cp_Grid)

# Check out the results:
rpart_cv_1
plot(rpart_cv_1)

# First, take the cp value of the best performing model in CV
tfidf_best_cp <- rpart_cv_1$bestTune$cp
# Then, extract performance measures for the best cp value 
tfidf_best_results <- rpart_cv_1$results %>% filter(cp==tfidf_best_cp)
tfidf_best_results

## Remainder:
## - sensistivity (true positive rate) = TP/(TP+FN)
##   (e.g., if positive class is passing an exam, sensitivity represents the percentage 
##    of students who passed the exam and were correctly identified as having a high 
##    chance of passing the exam)
## - specificity (true negative rate) = TN/(TN+FP)
##    (to continue with the example, specificity would be the percentage of students who 
##    failed the exam and were correctly identified as likely to fail) 

## Note that in this example we do not have 'true' positive and negative classes,
## that is, no preference for better predicting one class ('guns') over the other 
## ('mideast'). So, a balanced, good performance based on both sensitivity and 
## specificity is what we will be striving for. In other situations, this may not 
## be the case, and you'll put more focus on either sensitivity (focus on the 
## positive class) or specificity (focus on the negative class).
## For example, in case of spam detection, where 'spam' is the 'positive' class,
## one would put focus on maximizing specificity, that is, reducing the number of
## non-spam (aka 'ham') messages that were incorrectly classified as being spam.

# We can use the final model to examine feature importance
rpart_cv_1$finalModel$variable.importance %>%
  sort(decreasing = TRUE) %>%
#  names() %>%
  head(20)

# Another way to inspect relevant features
varImp(rpart_cv_1, useModel = FALSE) %>% plot(top=20)

# Note that these two ways of obtaining feature importance gave somewhat different results.
# The reason is that they apply different calculations to determine feature importance:
# - in rpart: the feature's overall contribution to the decrease in (node) impurity 
# - in caret: the overall reduction in the loss function attributed to a feature
# (for details check the documentation)

###########################################################
## USE UNIGRAMS AND BIGRAMS AND CHI2 FOR FEATURE SELECTION
###########################################################

# N-grams allow us to augment our DTM matrix with word ordering.
# This tends to lead to increased performance over ML models 
# trained with unigrams only. On the down side, the inclusion of
# ngrams, even only bigrams and trigrams, leads to an explosion 
# in the number of features. So, in addition to unigrams, 
# we will use only bigrams. 

# In addition, we will create the feature set in a different way,
# to demonstrate a bit of the quanteda's functions for creating and
# manipulating a corpus.

# Create a quanteda corpus out of the posts' content
train_corpus <- corpus(train_2cl$post_txt)
# Add newsgroup as a document level variable, to represent, for each
# post, the class the post belongs to
docvars(train_corpus, field = "newsgroup") <- train_2cl$newsgroup
# Get a summary for the first 10 documents in the corpus:
summary(train_corpus, n = 10)

# Starting from the corpus, create a dfm in a similar way 
# it was done before. Since the NUMBER feature did not prove 
# particularly important (see above the list of important features), 
# we can omit it, and simply remove numbers
train_dfm_2 <- train_corpus %>% 
  dfm(tolower = TRUE, 
      remove_punct = TRUE, 
      remove_symbols = TRUE,
      remove_numbers = TRUE,
      remove = stopwords("english"),
      stem = TRUE,
      ngrams = 1:2) %>%
  dfm_keep(min_nchar = 3)
train_dfm_2
# about 148K features

# We will use the Chi2 test of independence to do feature selection.
# The null hypothesis of the test is that the occurrence of a term 
# and the occurrence of a class label are independent; so, if a test
# proves to be significant, we reject the null hypothesis and 
# consider some dependence between the term and the class label.

# Since Chi2 is not reliable for low frequency terms,
# first, we will remove such terms and then compute Chi2

# Compute the overall (corpus) frequency for each term
# (note that values in the dfm are TFs)
dfm_2_tot_tf <- colSums(train_dfm_2)
# there is also quanteda's function textstat_frequency()
summary(dfm_2_tot_tf)
# Again, very (even more) uneven distribution
plot_word_weight_distr(wweights = dfm_2_tot_tf, 
                       lbl = "TF for unigrams and bigrams", 
                       bin_width = 100)
# Considering the (huge) number of terms and very low frequency of a large
# majority of them, keep only those with the overall (corpus) frequency 
# above the 75th percentile
to_keep <- which(dfm_2_tot_tf > quantile(dfm_2_tot_tf, probs = 0.75))
train_dfm_2 <- dfm_keep(train_dfm_2, 
                        pattern = names(dfm_2_tot_tf[to_keep]),
                        valuetype = "fixed", verbose = TRUE)
train_dfm_2
# reduced to ~34.16K features

# Next, we use chi2 to select the most discriminating features
chi2_vals <- dfm_group(train_dfm_2, "newsgroup") %>%
  textstat_keyness(measure = "chi2")
head(chi2_vals)
tail(chi2_vals)
# we're not interested in the sign, only in the strength of 
# the association; so, we'll take the absolute value of chi2
chi2_vals <- chi2_vals %>%
  mutate(chi2 = abs(chi2)) %>%
  arrange(desc(chi2))
head(chi2_vals, n=10)
tail(chi2_vals, n=10)

# To determine how to reduce the number of features based on the
# computed chi2 values, consider, first, the number of features that 
# are significantly associated with the class labels at alpha=0.05
chi2_vals %>%
  filter(p < 0.05) %>%
  nrow()
# still too high (7617)
# try with alpha=0.01
chi2_vals %>% filter(p < 0.01) %>% nrow()
# looks better: 3802 features

# Keep only the features with statistically significant association 
# with the class labels (alpha=0.01)
train_dfm_chi2 <- dfm_keep(train_dfm_2, 
                            pattern = chi2_vals$feature[chi2_vals$p < 0.01],
                            valuetype = "fixed", verbose = TRUE)
train_dfm_chi2
# reduced to ~3.8K features

# Create a data frame for training a classifier
train_df_2 <- create_feature_df(train_dfm = train_dfm_chi2,
                                class_labels = train_2cl$newsgroup)

###################################################################################
# BUILD the 2nd ML MODEL: RPART + CHI2-based FEATURE (UNIGRAMS & BIGRAMS) SELECTION
###################################################################################

# Build a CV-ed model (rpart classifier) with the new feature set and 
# all other settings unchanged
rpart_cv_2 <- cross_validate_classifier(seed, 
                                        nclust = 5,
                                        train_data = train_df_2,
                                        ml_method = "rpart",
                                        grid_spec = cp_Grid)

rpart_cv_2
plot(rpart_cv_2)

# First, take the cp value of the best performing model in CV
chi2_best_cp <- rpart_cv_2$bestTune$cp
# Then, extract performance measures for the best cp value 
chi2_best_results <- rpart_cv_2$results %>% filter(cp == chi2_best_cp)
chi2_best_results

# Compare the performance of the two classification models built so far
data.frame(rbind(tfidf_best_results, chi2_best_results), 
           row.names = c("TF_IDF", "TF_Chi2"))

# With the new model, we did better on all measures

# Before proceeding, let's examine the terms (features) used for building the model
rpart_cv_2$finalModel$variable.importance %>%
  sort(decreasing = TRUE) %>%
  names() %>% 
  head(20)

# and using caret's metric for variable importance:
varImp(rpart_cv_2) %>% plot(top=20)

# In the next step, we will apply a more sophisticated feature reduction method.
# In particular, we'll apply Singular Value Decomposition (SVD) to the DTM of
# TF-IDF weighted unigrams and bigrams.

####################################
# SINGULAR VALUE DECOMPOSITION (SVD)
# FOR REDUCING THE FEATURE SPACE
####################################

# We will now use Singular Value Decomposition (SVD) to reduce the number 
# of features (ngrams) to a significantly smaller set that explains 
# a large portion of variability in the data.

# Suggested reading for SVD and its use in text analysis 
# (Latent Semantic Analysis):
# - Landauer, T. K., Foltz, P. W., & Laham, D. (1998). Introduction to Latent 
#   Semantic Analysis. Discourse Processes, 25, 259-284. 
#   URL: http://lsa.colorado.edu/papers/dp1.LSAintro.pdf


# First, we should setup the data to which SVD will be applied.  
# What we need is TF-IDF weighted TDM (Term Document Matrix). 
# Note: TDM is nothing more than transposed DTM.
# train_dfm_2 will serve just well; we just need to substitute
# TF as the relevance metric (weight) with TF-IDF:
train_dfm_2 <- dfm_tfidf(train_dfm_2, scheme_tf = 'prop', scheme_df = 'inverse')

# Next, we need to set the number of the most important singular vectors we wish 
# to calculate and retain as features (in SVD terms, it is the rank the original 
# matrix is to be reduced to).
# How to determine the "right" number of singular vectors is still an open issue.
# Some useful links on that topic:
# - https://stackoverflow.com/questions/9582291/how-do-we-decide-the-number-of-dimensions-for-latent-semantic-analysis 
# - https://irthoughts.wordpress.com/2008/02/13/lsi-how-many-dimensions-to-keep/

# We will reduce the dimensionality down to 300 columns. This number is chosen as it
# is often recommended (based on the experience in practice).
# (NB. To get the best results, the number of dimensions would have to be  
# experimentally determined, by trying several different values and comparing 
# the performance of the resulting models)

# We'll use the *irlba* R package for SVD
library(irlba)
set.seed(seed)
svd_res <- irlba(t(as.matrix(train_dfm_2)), # SVD / LSA requires TDM (not DTM) as its input 
                 nv = 300, # the number of dimensions (singular vectors) to estimate
                 maxit = 600) # max iterations is set to be twice larger than nv 

# (n.b. the above function call takes about 6-8 min to execute)

# Examine the result:
str(svd_res)
# d - stores the singular values of the original matrix (values on the diagonal of the sigma matrix)
# u - its columns are referred to as left singular vectors of the original matrix; it represents the 
#     relation between the extracted dimensions (in the columns) and the ngrams (in the rows)
# v - its columns are referred to as right singular vectors of the original matrix; it respresents 
#     the relation between the extracted dimensions (columns) and the documents (rows)

# Store these vectors and matrices so that the computation 
# does not have to be repeated
saveRDS(svd_res$d, "models/svd/sigma.RData")
saveRDS(svd_res$u, "models/svd/left_sv.RData")
saveRDS(svd_res$v, "models/svd/right_sv.RData")

# Take a glimpse at the new feature set (the right singular vector):
View(svd_res$v[1:20,1:50])

# Create a new feature data frame using the 300 features obtained by applying
# SVD to TF-IDF weighted DTM (i.e. the V matrix produced by SVD)
train_svd_df <- cbind(Label = train_2cl$newsgroup, data.frame(svd_res$v))

# Next, we will examine the predictive power of the model with singular 
# vectors as features.

## Before proceeding to the creation of a classifier, note that there is
## an alternative approach to data preparation for SVD / LSA.
## It was suggested in the original paper on SVD / LSA by Landauer, Foltz, 
## & Laham (1998):
##    "Before the SVD is computed, it is customary in LSA to subject the data 
##    in the raw word-by-context matrix to a two-part transformation. 
##    First, the word frequency (+ 1) in each cell is converted to its log. 
##    Second, the information-theoretic measure, entropy, of each word 
##    is computed as: -p*logp over all entries in its row, 
##    and each cell entry then divided by the row [word] entropy value. 
##    The effect of this transformation is to weight each word-type occurrence  
##    directly by an estimate of its importance in the passage [document] and 
##    inversely with the degree to which knowing that a word occurs provides  
##    information about which passage [document] it appeared in."
##
## So, instead of TF-IDF, transform the original DTM (train_dfm_2) in the 
## manner suggested above, apply SVD on thus transformed DTM, and build 
## a RF model, as we do below. Compare the results with those of rf_cv_1 
## (given below). 

###############################################
# BUILD the 3rd ML MODEL: RANDOM FOREST + 
# SINGULAR VECTORS (FROM TF-IDF WEIGHTED DTM)
###############################################

# We have significantly reduced the dimensionality of our data using SVD. 
# Now, we can use a more complex and powerful classification algorithm. 
# In particular, we will build a Random Forest (RF) model.

## For a brief introduction to the Random Forest algorithm, 
## see the "Bagging and Random Forest" slides (made available as part of the WS materials).
## For more details and an excellent explanation of Random Forest and related algorithms,
## see chapter 8.2 of the Introduction to Statistical Learning book
## http://www-bcf.usc.edu/~gareth/ISL/ 

# We will build a RF model with 1000 trees. 
# We'll try different values of the mtry parameter to find the value that 
# leads to the best performance. 
# The mtry parameter stands for the number of features randomly sampled as 
# candidates at each split. 
# For the mtry parameter, we will consider 10 different values between the minimum
# (1 feature) and the maximum possible value (all features). 
n_features <- ncol(train_svd_df)-1
param_Grid <- expand.grid( .mtry = seq(from = 1, to = n_features, length.out = 10),
                         .splitrule = "gini", # gini is a measure of node 'purity'
                         .min.node.size = c(2,3)) 

# NOTE: The following code takes a long time to run. Here is why:
# We are performing 5-fold CV. That means we will examine each model configuration 
# 5 times. We have 20 configurations as we are asking caret to try 10 different
# values of the mtry parameter and 2 different values of the min.node.size parameter. 
# In addition, we are asking RF to build 1000 trees. Lastly, when the best values 
# for the parameters are chosen, caret will use them to build the final model using 
# all the training data. So, the number of trees we're building is:
# (5 * 20 * 1000) + 1000 = 1,001,000 trees!

# Build a RF classifier
rf_cv_1 <- cross_validate_classifier(seed, 
                                     nclust = 3,
                                     train_data = train_svd_df,
                                     ml_method = "ranger",
                                     grid_spec = param_Grid)

# (n.b. the above f. call takes about an hour to execute)

# Save the model to have a quick access to it later
saveRDS(rf_cv_1, "models/rf_cv_1.RData")

# Load the saved model
# rf_cv_1 <- readRDS("models/rf_cv_1.RData")

# Check out the results
rf_cv_1
plot(rf_cv_1)

# Extract evaluation measures for the best performing model 
svd_best_mtry <- rf_cv_1$bestTune$mtry
svd_best_min_nsize <- rf_cv_1$bestTune$min.node.size
svd_best_res <- rf_cv_1$results %>% 
  filter(mtry==svd_best_mtry & 
           min.node.size==svd_best_min_nsize)
svd_best_res

# Compare the results with the previously CV-ed models
comparison <- data.frame(rbind(tfidf_best_results %>% select(-cp), # exclude the cp parameter 
                               chi2_best_results %>% select(-cp), 
                               svd_best_res %>% select(-c(mtry:min.node.size))), # exclude the 3 model parameters 
                         row.names = c("RPART_TFIDF", "RPART_TF_CHI2", "RF_SVD"))
# Add a column with the number of features
comparison$NFeatures <- c(ncol(train_df),
                          ncol(train_df_2),
                          ncol(train_svd_df))
comparison
# The combined use of the new feature set and a more powerful algorithm significantly 
# improved the results, including the reduction in the variability of the results 
# (see SD values). In addition, the number of features is 10 - 12 times smaller than 
# in the other two models; this is highly important as it makes the model less prone 
# to overfitting.

# The downside is that the features are not interpretable - we can check for the most 
# important features, as we did before, but that will not be informative as the SVD 
# dimensions are rather obscure
plot(varImp(rf_cv_1), top = 20)


# Before moving to the evaluation of the model, we'll do a bit of cleaning
# to release the memory of the large objects that are no longer needed
remove(rpart_cv_1, rpart_cv_2, dfm_2_tot_tf, to_keep, train_tokens, 
       train_dfm_reduced, train_dfm_chi2, train_corpus)

##################
# TEST THE MODEL
##################

# Now, it is time to verify the model that proved the best  
# (the 3rd one: Singular Vectors + RF) using the test data 
# we set aside at the beginning of the script.  
# The first stage of the evaluation process is running the 
# test data through our text transformation pipeline of:
# - Tokenization (including removal of punctuation, symbols, and numbers)
# - Removing tokens less than 3 characters long
# - Lower casing
# - Stopword removal
# - Stemming
# - Adding bigrams
# - Creating DTM and ensuring the test DTM has 
#   the same features (ngrams) as the train DTM
# - Computing TF-IDF weights 
# - Feature set transformation / reduction using SVD  

test_tokens <- tokens(x = test_2cl$post_txt, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE,
                       remove_numbers = TRUE)

test_tokens <- tokens_keep(x = test_tokens, min_nchar = 3) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords()) %>%
  tokens_wordstem(language = "english") %>%
  tokens_ngrams(n = 1:2)
  
test_dfm <- dfm(test_tokens, tolower = FALSE)

# Compare the train and test DTMs
# (note: train DTM is the one that served as the input for 
# creating the features - SVD dimensions - of the best classifier)
train_dfm_2
test_dfm
# The two DTMs differ in the feature set. This is expected as features
# are ngrams from two different sets of posts (training and test).
# However, we have to ensure that the test DTM has the same n-grams 
# (features) as the training DTM.
# The rationale: we need to represent any new post in the feature space 
# that our classifier 'is aware of' (otherwise, it will report an error)
# and that is the feature space of the training set.

# Transform test_dfm so that it has the same features as the dfm that  
# was used to build features of our best classifier
test_dfm <- dfm_keep(test_dfm, pattern = train_dfm_2)
test_dfm
# Now, test dfm seems to have the same features as the train dfm.
# Let's check if those are really the same features
setdiff(featnames(test_dfm), featnames(train_dfm_2))
setdiff(featnames(train_dfm_2), featnames(test_dfm))
# No difference -> they are exactly the same.

# The next step is to 'project' the test DTM into the same 
# TF-IDF vector space we built for our training data. 
# This requires the following steps:
# 1 - Normalize term counts in each document (i.e. each row)
# 2 - Perform IDF multiplication using training IDF values
#
# NB. We'll use IDF values computed on the training set, since 
# IDF is always computed on a representative and sufficiently large
# corpus, and in production settings (when the classifier is deployed),
# we won't have sufficiently large number of unclassified posts to use
# for IDF computation. Hence, the training set is used as a representative,
# large corpus; altenatively, we may use some other large corpus from 
# the same domain and of the same writing style (e.g. forum posts)

# Normalize term counts in all test posts
test_tf <- dfm_weight(test_dfm, scheme = "prop")

# Next, compute IDF values for the features using the training set
# (Note: the inverse_doc_freq() f. is defined in the tm_utils.R script)
train_idf <- apply(train_dfm_2, 2, inverse_doc_freq)

# Next, calculate TF-IDF using the computed IDF values
test_tfidf <-  apply(test_tf, 1, function(x) x*train_idf)
dim(test_tfidf)
# Note that documents are given in columns, while ngrams are in rows,
# meaning this is TDM; we will keep it this way as this form will be
# required in the next step

# With the test data projected into the TF-IDF vector space of the training
# data, we can now do the final projection into the training SVD space
# (i.e. apply the SVD matrix factorization).

##############################################
## APPLYING SVD PROJECTION ON A NEW DATA SET
##############################################

# The formula for projecting a particular document (d) to the SVD space: 
#
# d_hat = sigma_inverse * transposed_U_matrix %*% d_TF-IDF_vector
#
# d_hat is the representation of the given document d in the SVD space of 
# the training dataset; more precisely, it is the representation of d in
# terms of the dimensions of the V matrix (right singular vectors).
# 
# Before applying this formula, let us examine why and how do we use it

# To demonstrate that the above formula will allow us to represent documents
# using the SVD dimensions, we'll examine how it can be used on a document 
# from the training set, since for documents from the training set we know 
# what SVD representation looks like (it is given by the V matrix)
example_doc <- as.matrix(train_dfm_2)[1,]

# For convenience, we'll introduce:
sigma_inverse <- 1 / svd_res$d # 1 / readRDS("models/svd/sigma.Rdata") 
u_transpose <- t(svd_res$u) # readRDS("models/svd/left_sv.Rdata") %>% t() 

# The projection of the example document in the SVD space, based on the 
# above given formula:
example_doc_hat <- as.vector(sigma_inverse * u_transpose %*% example_doc)
# Look at the first 10 components of projected document...
example_doc_hat[1:10]
# ... and the corresponding row in the document space produced by SVD (the V matrix)
svd_res$v[1, 1:10]
# v <- readRDS("models/svd/right_sv.Rdata")
# v[1,1:10]
# The two vectors are almost identical (note the values are expressed in e-04, e-05,...).
# In fact, the differences are so tiny that when we compute cosine similarity 
# between the two vectors, the similarity turns to be equal to 1:
library(lsa)
cosine(example_doc_hat, svd_res$v[1,]) # v[1,]
#
# Why is this useful?
# It shows that using the above given formula, we can transform any document into
# the singular vector space of the training set, using the computed sigma_inverse 
# and transposed_U_matrix; this further means that we can take a new, unseen 
# document (a post in our case), represent it as a TF-IDF weighted vector, and 
# transform it into the singular vector space so that it can be classified by our 
# prediction model.


# So, we will use the above given formula to represent posts from the test set in 
# the singular vector space. As we have multiple documents, we need to replace 
# d_TF-IDF_vector (3rd element on the left), with a matrix of TF-IDF values 
# (the matrix should have terms in rows and documents in columns)
test_svd_hat <- sigma_inverse * u_transpose %*% test_tfidf
dim(test_svd_hat)

###################################################
## MAKE PREDICTIONS ON THE (TRANSFORMED) TEST DATA
###################################################

# With the feature set ready, we can now build the test data frame to 
# feed into our prediction model
test_svd_df <- data.frame(Label = test_2cl$newsgroup, 
                          t(test_svd_hat)) # need to transpose it, to place documents in rows

# Now we can make predictions on the test data set 
# using our best classifer (rf_cv_1)
rf_cv_1 <- readRDS("models/rf_cv_1.Rdata")
preds <- predict(rf_cv_1, newdata = test_svd_df)

# Examine the results
# 1) inspect confusion matrix
cm <- table(Actual = test_svd_df$Label, Predicted = preds)
cm

# 2) compute evaluation measures
# (note that the get_eval_measures() function is defined in tm_utils.R)
eval_metrics <- c('Sensitivity', 'Specificity', 'AUC')
get_eval_measures(rf_cv_1, test_svd_df, eval_metrics)

# Let's compare these results with those obtained on the training data 
svd_best_res[, c('Sens', 'Spec', 'ROC')]

# The performance is lower than on the training set, which is expected
# (one should almost always expect lower performance on the test set)
# However, it is still rather good.

