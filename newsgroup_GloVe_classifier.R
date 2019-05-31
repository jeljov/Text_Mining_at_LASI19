## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'19 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi19/lasi19-workshops-tutorials/)


## The script provides an example of the text classification task using a subset of  
## the 20 Newsgroups dataset and word vectors derived from a pre-trained GloVe model.
## For a brief exploration of a GloVe model, see the 'exploring_word_vectors' script.

## In this script, we will examine some simple ways of using pre-trained word vectors 
## to create document features that can be subsequently used for the classification
## task. These consist of combining, in different ways, word vectors of the words 
## a document consists of; often used combinations include:
## - computing the average of the document's word vectors 
## - computing weighted average of the document's word vectors
## - taking min and max values of weighted word vectors and concatenating them.
## These methods are found to perform well on short and topically coherent texts. 
## This means that they can be expected to perform well on posts exchanged in 
## online communication channels that are often used in e-learning contexts 
## (e.g. forums, chats, Twitter, and other similar forms of online social networks).
##
## We will also examine the use of Word Mover Distance (WMD) for computing document 
## similarity by leveraging vector representation (word vectors / embeddings) of
## words the documents consist of. The computed document similarities are then used as
## the input for the K Nearest Neighbours (kNN) classification algorithm. This method
## is also found to perform well on short texts.

## For the document classification task, we will use the same subset of the 
## 20 Newsgroups dataset as the one used in the 'newsgroup_classifier' R script.
## In particular, the subset includes the newsgroup on political issues in the 
## Middle East (talk.politics.mideast) and the one on the legality of the use of 
## guns (talk.politics.guns). This way we will be able to compare the performance 
## of word vector based models with those built using more "traditional" features.


######################
# SET UP the SESSION
######################

# Load the required libraries
# (more libraries will be loaded as they become needed)
library(tidyr)
library(dplyr)
library(readr)
library(quanteda)

# Load auxiliary functions 
source("tm_utils.R")
source("word_vec_utils.R")

# Set the seed to assure reproducibility of the results
seed <- 8219

####################################
#
# LOAD THE DATA AND CREATE A CORPUS
#
####################################

# Since we have already done the pre-processing of the 20 Newsgroups dataset,
# we will use the pre-processed data
# (for pre-processing details see the 'preprocess_20News_dataset' R script)
usenet_data <- read_csv(file = "data/20news/20news-bydate-train.csv") %>%
  data.frame()
glimpse(usenet_data)

# Select groups on mideast (talk.politics.mideast) and guns (talk.politics.guns)
usenet_data <- usenet_data %>% 
  filter(newsgroup %in% c('talk.politics.guns', 'talk.politics.mideast'))

# Since the approach we intend to apply is suitable for shorter texts,
# we should check the length of the selected posts and remove overly long ones. 
# An easy way to do that is to rely on the readily available statistics for
# quanteda corpus. So, we'll start by creating a quanteda corpus out of the 
# selected newsgroups posts:
usenet_corpus <- corpus(usenet_data$post_txt)
# Add newsgroup and post id as document level variables
docvars(usenet_corpus, field = "newsgroup") <- usenet_data$newsgroup
docvars(usenet_corpus, field = "post_id") <- usenet_data$id

# Compute some basic statistics about the token counts per newsgroup
# to get an idea about the distribution of the post length:
usenet_corpus %>% 
  summary(n = nrow(usenet_data)) %>% # include all posts in the summary
  group_by(newsgroup) %>%
  summarize(doc_cnt = n(),
            avg_token = mean(Tokens),
            median_token = median(Tokens),
            Q3_token = quantile(Tokens, probs = 0.75),
            max_token = max(Tokens))

# According to the examined token-related statistics, the two selected groups 
# have some overly long posts. We will check this more closely and remove 
# posts that are excessively long. 

# Start by computing summary statistics for each document (post) in the corpus
corpus_stats <- usenet_corpus %>% 
  summary(n=nrow(usenet_corpus$documents))
glimpse(corpus_stats)

# Examine the distribution of post length (= token count) visually
# (plot_ng_feature_comparison() is defined in the tm_utils.R script)
plot_ng_feature_comparison(corpus_stats, 'Tokens', 'Token count')
# A lot of outliers... Let's examine them closer
sort(boxplot.stats(corpus_stats$Tokens)$out, decreasing = TRUE)
# 1500 tokens seems to be a reasonable threshold: 
# - 40 (3.62%) posts will be removed 
# - texts of up to 1500 tokens (tokens are not only words but also punctuation marks, symbols
#   numbers...), while not short, may be still acceptable, especially as being forum posts 
#   they are expected to be topically coherent

# Removing, from the corpus, posts (documents) with more than 1500 tokens
posts_to_remove <- corpus_stats %>% filter(Tokens > 1500) %>% pull(post_id)
usenet_corpus <- corpus_subset(usenet_corpus, 
                               subset = !post_id %in% posts_to_remove)


########################################
#
# USE GLOVE MODEL FOR FEATURE CREATION 
#
########################################

# Extract tokens from the corpus and filter out all those tokens that are
# not expected to be useful for classification purposes. The rationale:
# since we will do the averaging over word vectors within a document, we
# should try to assure that only semantics-bearing words are kept.

post_tokens <- tokens(x = usenet_corpus, 
                      what = "word", 
                      remove_numbers = TRUE, 
                      remove_punct = TRUE,
                      remove_symbols = TRUE,
                      remove_twitter = TRUE, # removes leading '#' and '@' characters
                      remove_url = TRUE)

# Normalize tokens (set them to lower case), remove stopwords and 
# tokens with only 1 or 2 letters
post_tokens <- post_tokens %>%
  tokens_tolower() %>%
  tokens_remove(stopwords()) %>%
  tokens_keep(min_nchar = 3) 

# Note that we are NOT stemming the tokens since words in the GloVe 
# model were not stemmed, and we need to match against those words.

# Create DTM
post_dtm <- dfm(post_tokens, tolower=FALSE)
post_dtm

# Extract words (features) from the DTM since we need to match  
# these against the words in the pre-trained GloVe model
post_words <- featnames(post_dtm)
# ... and examine them
head(post_words, n = 100)
tail(post_words, n = 100)


# Load the pre-trained GloVe word vectors
#
# In particular, we will use the same GloVe model as in 
# the "exploring_word_vectors" script where we introduced GloVe
# (it is the 'glove.6B.300d' model with 300 dimension vectors)

# Read in the the model 
# (Note: change the 'glove_6B_300d_dir' variable to the path of the directory 
#  where the "glove.6B.300d.txt" file is stored on your computer)
glove_6B_300d_dir <- "C:\\Users\\jovanje\\Documents\\LASI 2019\\glove.6B\\"
#glove_6B_300d_dir <- "~/R Studio Projects/Large datasets/glove.6B/"
g6b_300d <- scan(file = paste0(glove_6B_300d_dir, 'glove.6B.300d.txt'), what="", sep="\n")

# Create a data frame out of the large vector read from the file
# (get_word_vectors_df() is defined in the tm_utils.R script)
g6b_300d_df <- get_word_vectors_df(g6b_300d, verbose = TRUE)
# Remove g6b_300d to release memory
remove(g6b_300d)

# Take the words from the GloVe model - we need these words to 
# match them against the features (words) from the corpus DTM
glove_words <- colnames(g6b_300d_df)


# The next step is to match words from the post_dtm to
# the corresponding word vectors in the loaded GloVe model,
# and keep only those words that are present both in 
# post_dtm and in the GloVe model. 
words_to_keep <- intersect(post_words, glove_words)
# check the 'level' of matching
length(words_to_keep)/length(post_words)
# 13795 (83.06%) words from our DTM have their vectors in GloVe

# Let's briefly inspect words from post_dtm that are not in GloVe
setdiff(post_words, glove_words)[1:100]
# Mostly abbreviations, misspelled words, and compound words
# However, there are also 'regular' words in the posessive form -
# e.g. vancouver's, everyone's, criminal's, etc.
# We can improve the matching level if we substitute these posessive
# forms with their 'regular' counterparts (eg. one's -> one)
tokens_to_replace <- tokens_keep(x = post_tokens,
                                 pattern = "[a-z]+'s",
                                 valuetype = "regex", verbose = TRUE)
tokens_to_replace <- unlist(tokens_to_replace) %>% unique()
replacements <- gsub(pattern = "([a-z]+)'s", replacement = "\\1", 
                     x = tokens_to_replace)
# Now, re-create DTM
post_dtm <- dfm_replace(post_dtm, 
                        pattern = tokens_to_replace,
                        replacement = replacements, verbose = TRUE)

# Again, get the words that are present both in the DTM and GloVe model
words_to_keep <- intersect(featnames(post_dtm), glove_words)
length(words_to_keep)/length(post_words)
# 13840 (83.33%) - a slight improvement

# Create a new DTM that will keep only those words (columns)
# from the original DTM (post_dtm) that are present in the GloVe model  
dtm_reduced <- dfm_keep(post_dtm, 
                        pattern=words_to_keep, 
                        valuetype="fixed", 
                        verbose=TRUE)

# Likewise, from GloVe, select word vectors that will be used for building 
# a feature set, that is, vectors of the words present in the dtm_reduced
glove_to_keep_indices <- which(glove_words %in% words_to_keep)
g6b_300d_df_reduced <- g6b_300d_df[,glove_to_keep_indices]
# Remove the original glove df (g6b_300d_df)
remove(g6b_300d_df)

# Order the columns (words) in the g6b_300d_df_reduced, to be the same as in
# the dtm_reduced
g6b_300d_df_reduced <- g6b_300d_df_reduced[,colnames(dtm_reduced)]

# Before proceeding, remove large objects that are no longer needed
remove(usenet_data, corpus_stats, post_tokens, glove_words, glove_to_keep_indices,
       post_words, post_dtm, words_to_keep, glove_6B_300d_dir)

################################################
# 
# CREATE DOCUMENT FEATURE MATRIX 
# BY COMPUTING WEIGHTED AVERAGE OF WORD VECTORS
# (WEIGHTS: TF-IDF)
# 
################################################

# Compute feature values for each post as the (coordinate-wise) TF-weighted mean 
# value across all the word vectors.
#
# Note that after the above reduction of DTM and GloVe to the common set of 
# features (words), the two matrices have the same number of columns, and also the
# same column headers (words).
# Now, we will take each post (row) from the DTM and multiply it with the transposed 
# GloVe matrix, thus, in fact weighting word vectors in GloVe with the post-specific 
# TF-IDF weights of the corresponding words. As the result, we will get a matrix of
# TF-IDF weighted word vectors (words in rows, dimensions in columns) for each post. 
# Next, we take the mean value (across words) for each dimension (column), to obtain
# a new feature vector for each post; these vectors have the same number of features 
# as there are dimensions in the GloVe model (300). This way, we are, in fact,  
# translating the existing feature space (words in DTM) into a new feature space 
# (dimensions of the GloVe word vectors).

# First, we need to compute TF-IDF weights for the words in the DTM 
tf_idf_dtm <- dfm_tfidf(dtm_reduced, 
                        scheme_tf = "prop") # for TF, use normalized counts (ie. proportions)

# Since we will need to build document features, now for the training set, and 
# later on for the test set, it is better to pack the code into a function
build_word_vec_features <- function(dtm, word_vec_df) {
  word_vec_feats <- data.frame()
  for(i in 1:nrow(dtm)) {
    doc <- as.matrix(dtm)[i,]  
    doc_word_vecs <- doc * t(word_vec_df) 
    doc_features <- apply(doc_word_vecs, 2, mean)  
    word_vec_feats <- as.data.frame(rbind(word_vec_feats, doc_features))
  }
  colnames(word_vec_feats) <- paste0("dim_",1:ncol(word_vec_feats))
  word_vec_feats
} 

word_vec_features <- build_word_vec_features(tf_idf_dtm, g6b_300d_df_reduced)
dim(word_vec_features)

# Add class label to the feature matrix
lbl <- ifelse(docvars(usenet_corpus, field = "newsgroup") == "talk.politics.mideast",
              yes = "mideast", no = "guns")
word_vec_features$Label <- as.factor(lbl)

# Check the class proportion
table(word_vec_features$Label)
# almost perfect balance!


##########################################
# 
# USE RANDOM FOREST TO BUILD A CLASSIFIER
# THROUGH CROSS-VALIDATION
# 
##########################################

# Create a Random Forest (RF) classifier.
# The training steps are exactly the same as those applied when creating 
# a RF classifier in  the 'newsgroups_classifier' R script. 
n_features <- ncol(word_vec_features) - 1
mtry_Grid = expand.grid( .mtry = seq(from = 1, to = n_features, length.out = 10),
                         .splitrule = "gini", 
                         .min.node.size = c(2,3))
rf_cv_1 <- cross_validate_classifier(seed, 
                                     nclust = 5,
                                     train_data = word_vec_features,
                                     ml_method = "ranger",
                                     grid_spec = mtry_Grid)

# Save the model to have it available for later
saveRDS(rf_cv_1, "models/glove/rf_cv_1.RData")

# Load the saved model
# rf_cv_1 <- readRDS("models/glove/rf_cv_1.RData")

# Check out the CV results
rf_cv_1
plot(rf_cv_1)

# Examine the model's performance in more detail
best_mtry <- rf_cv_1$bestTune$mtry
best_min_nsize <- rf_cv_1$bestTune$min.node.size
best_res <- rf_cv_1$results %>% 
  filter(mtry==best_mtry & min.node.size==best_min_nsize)
best_res %>% select(ROC:SpecSD)

# Excellent performance, very close to the performance of the best model in the
# newsgroups_classifier.R script.


######################################
#
# TEST THE MODEL (GLOVE VECTORS + RF)
#
######################################

# To test the classifier, we will apply a procedure that is highly similar to 
# the one we used in the 'newsgroups_classifier' R script. So, for the details
# of the procedure, please, refer to the other script; this script covers in
# detail only those elements of the procedure that differ from the 
# newsgroups_classifier.R


# We'll start by loading the test data
test_data <- read_csv(file = "data/20news/20news-bydate-test.csv") %>%
  data.frame()
# Select groups on mideast (talk.politics.mideast) and guns (talk.politics.guns)
test_data <- test_data %>% 
  filter(newsgroup %in% c('talk.politics.guns', 'talk.politics.mideast'))


# Next, transform the test data into a feature set

# To prepare the test set, we will follow the same steps as in 'newsgroups_classifier.R' 
# to transform the test data (ie. test posts) into a TF_IDF weighted document feature 
# matrix (dfm). The difference is in how this TF-IDF weighted dfm is further transformed:
# in the 'newsgroups_classifier' R script, we used the SVD-based transformation,
# whereas here, we will use GloVe word vectors.

test_tokens <- tokens(x = test_data$post_txt, 
                      what = "word", 
                      remove_numbers = TRUE, 
                      remove_punct = TRUE,
                      remove_symbols = TRUE,
                      remove_twitter = TRUE, 
                      remove_url = TRUE)

test_tokens <- tokens_keep(x = test_tokens, min_nchar = 3) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords())

test_tf_dfm <- dfm(test_tokens, tolower = FALSE) 
test_tf_dfm

# Transform test_dfm so that it has the same features (words) as 
# the dfm that was used to build features for the classifier
# (the one obtained by matching words from the training set and 
# the pre-trained GloVe model)
test_tf_dfm <- dfm_keep(test_tf_dfm, pattern = tf_idf_dtm)
test_tf_dfm

# Compute IDF values for the features using the training set
# (reminder: the inverse_doc_freq() f. is defined in the tm_utils.R script)
train_idf <- apply(tf_idf_dtm, 2, inverse_doc_freq)

# Next, calculate TF-IDF using the computed IDF values; normalize TF scores
# before multiplying them with IDF values
test_tfidf_dtm <- test_tf_dfm %>%
  dfm_weight(scheme = "prop") %>%
  apply(., 1, function(x) x*train_idf)
dim(test_tfidf_dtm)

# Transpose the matrix so that documents are given in rows
test_tfidf_dtm <- t(test_tfidf_dtm)

# Transform test_tfidf_dtm into the vector space of the GloVE word vectors
test_features <- build_word_vec_features(test_tfidf_dtm, g6b_300d_df_reduced)

# Make predictions on the (transformed) test data

# With the feature set ready, we can now build the test data frame to 
# feed into our prediction model
test_lbl <- factor(test_data$newsgroup, 
                   levels = unique(test_data$newsgroup),
                   labels = c("guns", "mideast"))
test_df <- data.frame(Label = test_lbl, test_features) 

# Now we can make predictions on the test data set 
# using the built classifer (rf_cv_1)
preds <- predict(rf_cv_1, newdata = test_df)

# Examine the results
# 1) inspect confusion matrix
cm <- table(Actual = test_df$Label, Predicted = preds)
cm

# 2) compute evaluation measures
eval_metrics <- c('Sensitivity', 'Specificity', 'AUC')
get_eval_measures(rf_cv_1, test_df, eval_metrics)

# Compare the test results with those obtained on the training data 
best_res[, c('Sens', 'Spec', 'ROC')]

# The performance is somewhat weaker than on the training data set, 
# especially, in terms of specificity, but overall it is still rather good.   

# Before proceeding, free up some memory by removing large objects that are 
# no longer needed
remove(usenet_corpus, test_tokens, train_idf)

##########################################
# 
# BUILD A CLASSIFIER BASED ON: 
# - GLOVE WORD VECTORS
# - (RELAXED) WORD MOVER DISTANCE 
# - K NEAREST NEIGHBOURS (KNN) ALGORITHM
# 
##########################################

# Now, we will create another classifier based on: 
# - (Relaxed) Word Mover Distance (WMD) for computing the distance / 
#   dissimilarity of documents (posts) 
# - kNN algorithm for classifying documents based on the WMD distances

# WMD and Relaxed WMD are introduced in the paper:
# Kusner, M. J., Sun, Y., Kolkin, N. I., & Weinberger, K. Q. (2015). 
# From Word Embeddings to Document Distances. In Proc. of the 32Nd Int'l Conf. 
# on Machine Learning - Vol. 37 (pp. 957–966). Lille, France: JMLR.org.
# URL: http://proceedings.mlr.press/v37/kusnerb15.pdf
#
# Check Figure 1 (p.1) in the paper to quickly get an intuition (general idea)
# this metric is based upon.

# We'll start by computing Relaxed WMD between each document pair using the 
# appropriate functions from the *text2vec* R package
library(text2vec)

# Create a Relaxed WMD (RWMD) object by specifying 2 input parameters:
# - word vector matrix with words given in rows and dimensions of the 
#   embedding space in columns; rows should have word names.
# - the method to be used for computing the distance between word vectors
rwmd_model = RWMD$new(wv = t(g6b_300d_df_reduced), method = "cosine")

# NB: in the original paper on WMD, authors use Euclidean distance when computing
# distance between word vector pairs; however, the author of text2vec suggests using
# Cosine similarity as in their experience, it results in better performance.
# In that case, distance is computed as: 1 - cosine_between_wv

# Now, we use the RWMD object to compute distances between each document pair.
# First, compute the RWMD distances between each pair of documents (posts) 
# within the training set: 
rwmd_dist_train_set <- dist2(x = dtm_reduced, method = rwmd_model, norm = "none") 
dim(rwmd_dist_train_set)

# NB: note that in the call of the dist2 function, norm parameter is set to 'none'
# this is done as RWMD can be computed only on the raw word counts

# Next, do the same but for train-test pairs, that is, for each document in the 
# test set compute its distance from the document in the train set (required for 
# the kNN classifier)
rwmd_dist_train_test <- dist2(x = test_tf_dfm, y = dtm_reduced,
                              method = rwmd_model, norm = 'none')
dim(rwmd_dist_train_test)


# To build a KNN classifier, we will use the *FastKNN* R package.
# The reason for choosing this package is that it allows for building a KNN classifier
# using precomputed distances, which is not the case with the often used knn() f. from 
# the class package (and many other packages).
library(FastKNN)

# Use the training dataset to find the best value for K

# We will run the kNN algorithm with a range of (odd) values for K,
# compute evaluation metrics for each K value and eventually
# choose the value that maximizes the evaluation metrics.
train_labels <- word_vec_features$Label
knn_eval_df <- data.frame()
eval_metrics <- c("Sensitivity", "Specificity", "F1", "Accuracy", "Kappa")
for(k in seq(from = 5, to = 35, by=2)) {
  set.seed(seed)
  knn_res <- knn_training_function(dataset = dtm_reduced,
                                   distance = rwmd_dist_train_set,
                                   label = train_labels,
                                   k = k)
  knn_eval <- confusionMatrix(data = as.factor(knn_res), reference = train_labels)
  knn_eval <- extract_eval_measures(knn_eval, eval_metrics)
  knn_eval <- c(k, knn_eval)
  knn_eval_df <- as.data.frame(rbind(knn_eval, knn_eval_df))
}
colnames(knn_eval_df) <- c("K", "Sensitivity", "Specificity", "F1", "Accuracy", "Kappa")
knn_eval_df
# Sort the results based on the F1 measure
arrange(knn_eval_df, desc(F1))
# k=5 is the best k value based on all the metrics except sensitivity 
# It is also associated with a low risk of overfitting 
# (the higher the value for k, the higher susceptibility to overfitting).

# Now, evaluate the model, with k=5, on the test set
knn_pred <- knn_test_function(dataset = dtm_reduced,
                              test = test_tf_dfm,
                              distance = rwmd_dist_train_test,
                              labels = train_labels,
                              k = 5)
# Use the computed predictions and the test set labels to evaluate the model
knn_eval <- confusionMatrix(data = as.factor(knn_pred), 
                            reference = test_df$Label)
# (note: the extract_eval_measures() f. is defined in the tm_utils script)
extract_eval_measures(knn_eval, eval_metrics)

# Let's compare these results with those obtained with the RF model
# with weighted average of word vectors
get_eval_measures(rf_cv_1, test_df, eval_metrics)

# The comparison suggests that this model (RWMD + kNN) is somewhat 
# weaker than the one based on weighted average of word vectors and RF
# (according to all examined metrics except sensitivity), but still of
# comparable performance. 


# Suggestion: you may want to check the following article:
# http://xplordat.com/2018/10/09/word-embeddings-and-document-vectors-part-2-classification/ 
# since it presents a well done comparison of classifiers built for the 20 Newsgroups  
# multiclass classification task, using: 
# - traditional document vectors (ie. bag-of-words model) vs word vectors, 
# - different word embeddings: word2vec, GloVe, and FastText; both pre-trained and custom built
# - different options for text preprocessing
# - different classification algorithms: Naive Bayes, Support Vector Machines (w/ linear kernel), 
#   and Neural Net (Multi-layer Perceptron)
# The article also provides some useful general take-aways. 
