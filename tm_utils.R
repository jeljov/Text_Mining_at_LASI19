## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'19 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi19/lasi19-workshops-tutorials/)


## The function reads all files from the given folder (infolder) 
## into a data frame and returns the created data frame
read_folder <- function(infolder) {
  tibble(file = dir(infolder, full.names = TRUE)) %>%
    mutate(content = map(file, read_lines)) %>%
    transmute(id = basename(file), content) %>%
    unnest(content)  # content is a list-column; unnest transforms each element of the list into a row
}


## For each post in the input data frame, the function removes the header and the automated 
## signature. It returns a data frame of the same structure, but with cleaned post text.
## Note: the 1st cumsum() removes all lines before an 'empty' line;
## since, in most of the posts, an empty line delimits the header from the rest of 
## the post content, this way, the header will be removed.    
## The 2nd cumsum() keeps all the lines until the line starting with one or more hyphens ('-')
## This way, the part of the post with automatic signature is removed.
remove_header_and_signature <- function(newsgroup_df) {
  newsgroup_df %>%
    group_by(newsgroup, id) %>%  # each group contains lines of text from one post
    filter(cumsum(content == "") > 0, 
           cumsum(str_detect(content, "^-+")) == 0) %>%
    ungroup()
}


## The function does further cleaning of the post content. In particular, 
## it removes lines that 
## - are empty, or 
## - start with "In article <", or 
## - end with "writes:" or "writes..."
## The function also removes the "quotation" characters (">" and "}") from 
## lines that start with 1 or more of such characters
clean_quoted_text <- function(newsgroup_df) {
  newsgroup_df %>%
    filter(!str_detect(content, "^In (article)? <.*")) %>% 
    filter(!str_detect(content, "(writes|wrote|says)(:|\\.\\.\\.)$")) %>% 
    mutate(content = str_replace(content, "^([>|\\}]+)\\s*([A-Za-z0-9]*.*)", "\\2")) %>%
    filter(content != "")
}


## The function transforms the input data frame by merging all pieces of text 
## that belong to one post and associating the merged content with the post id
# Note: since posts with the same id may appear in different newsgroups, to have
# a unique post identifier (post_id), we need to combine 'newsgroup' and 'id' 
# variables. After the post content is merged, we split the post_id back to 
# 'newsgroup' and 'id' variables (the last line)
merge_post_text <- function(newsgroup_df) {
  newsgroup_df %>%
    transmute(post_id = str_c(newsgroup, id, sep = "_"), content) %>%
    group_by(post_id) %>%
    summarise(post_txt = str_c(content, collapse = " ")) %>%
    ungroup() %>%
    separate(col = post_id, into = c("newsgroup", "id"), sep = "_") 
}


## Function for creating a feature data frame out of
## - a DTM, represented in the form of quanteda's dfm, and 
## - a vector of class labels
create_feature_df <- function(train_dfm, class_labels) {
  
  train_df <- convert(train_dfm, "data.frame")
  # The 'convert' f. from quanteda adds 'document' as the 1st feature (column)
  # in the resulting data frame. It needs to be removed before the data frame 
  # is used for training.
  if ((names(train_df)[1] == 'document') & (class(train_df[,1])=='character'))
    train_df <- train_df[, -1]
  
  # Check if there are documents that have 'lost' all their words, that is,
  # if there are rows with all zeros
  doc_word_cnt <- rowSums(train_df)
  zero_word_docs <- which(doc_word_cnt == 0)
  # If there are zero-word rows, remove them
  if (length(zero_word_docs) > 0) {
    print(paste("Number of documents to remove due to sparsity:", length(zero_word_docs)))
    train_df <- train_df[-zero_word_docs,]
    class_labels <- class_labels[-zero_word_docs]
  }
  
  # Assure that column names are regular R names
  require(janitor)
  train_df <- clean_names(train_df)
  
  # Combine class labels and the features 
  cbind(Label = class_labels, train_df)
  
}

## Function for plotting the distribution of word weights
plot_word_weight_distr <- function(wweights, lbl, bin_width = 0.1) {
  require(ggplot2)
  ggplot(data = data.frame(weights = wweights), mapping = aes(x = weights)) + 
    geom_histogram(aes(y=..density..),  # Histogram with density instead of count on y-axis
                   binwidth=bin_width,
                   colour="black", fill="grey") +    
    geom_density(alpha=.2, size=1) +
    xlab(lbl) +
    theme_bw() 
}


## The function uses boxplots to plot the distribution for the
## given feature; a separate boxplot is drawn for each newsgroup
plot_ng_feature_comparison <- function(df, feature, f_name) {
  require(ggplot2)
  ggplot(mapping = aes(x = df[['newsgroup']], 
                       y = df[[feature]], 
                       fill = df[['newsgroup']])) +
    geom_boxplot() +
    labs(x = "Newsgroups", y = f_name) +
    scale_fill_discrete(name="Newsgroups") +
    theme_bw()
}


## Function for performing 5-fold cross validation on the given training data set
## (train_data) using the specified ML algorithm (ml_method). 
## Cross-validation is done in parallel on the specified number (nclust) of logical cores.
## The grid_spec serves for passing the grid of values to be used in tuning one or more 
## parameter(s) of the ML method.
## The ntree parameter can be used to set the number of trees when Random Forest is used.
cross_validate_classifier <- function(seed,
                                      nclust, 
                                      train_data, 
                                      ml_method,
                                      grid_spec,
                                      ntree = 1000) { 
  require(caret)
  require(doSNOW)
  
  # Setup the CV parameters
  cv_cntrl <- trainControl(method = "cv", 
                           number = 5, 
                           search = "grid",
                           summaryFunction = twoClassSummary, # computes sensitivity, specificity, AUC
                           classProbs = TRUE, # required for the twoClassSummary f.
                           allowParallel = TRUE) # default value; set here to emphasize the use of parallelization
  
  # Create a cluster to work on nclust logical cores;
  # what it means (simplified): create nclust instances of RStudio and 
  # let caret use them for the processing 
  cl <- makeCluster(nclust, 
                    type = "SOCK") # SOCK stands for socket cluster
  registerDoSNOW(cl)
  
  # Track the time of the code execution
  start_time <- Sys.time()
  
  set.seed(seed)
  if (ml_method=="rpart")
    model_cv <- train(x = train_data %>% select(-Label),
                      y = train_data$Label,
                      method = 'rpart', 
                      trControl = cv_cntrl, 
                      tuneGrid = grid_spec, 
                      metric = 'ROC')
  if (ml_method=="ranger") {
    require(ranger)
    model_cv <- train(x = train_data %>% select(-Label),
                      y = train_data$Label,
                      method = 'ranger', 
                      trControl = cv_cntrl, 
                      tuneGrid = grid_spec, 
                      metric = 'ROC',
                      num.trees = ntree,
                      importance = 'impurity',
                      verbose = TRUE)
  }
  
  # Processing is done, stop the cluster
  stopCluster(cl)
  
  # Compute and print the total time of execution
  print(difftime(Sys.time(), start_time, units = 'mins'))
  
  # Return the built model
  model_cv
  
}

## Function for calculating relative (normalized) term frequency (TF)
relative_term_frequency <- function(row) { # in DTM, each row corresponds to one document 
  row / sum(row)
}

## Function for calculating inverse document frequency (IDF)
## Formula: log(corpus.size/doc.with.term.count)
inverse_doc_freq <- function(col) { # in DTM, each column corresponds to one term (feature) 
  corpus.size <- length(col) # the length of a column is in fact the number of rows (documents) in DTM
  doc.count <- length(which(col > 0)) # number of documents that contain the term
  log10(corpus.size / doc.count)
}

## Function for calculating TF-IDF
tf_idf <- function(x, idf) {
  x * idf
}


## The function first creates a model evaluation object by
## invoking the confusionMatrix() f. of the caret package, and 
## passes this object to the next function that extracts the
## requested evaluation measures
get_eval_measures <- function(pred_model, test_df, metrics) {
  require(caret)
  model_eval <- confusionMatrix(data = predict(pred_model, test_df), 
                                reference = test_df$Label)
  
  eval_measures <- extract_eval_measures(model_eval, metrics)
  
  if(('AUC' %in% metrics) | ('ROC' %in% metrics)) {
    eval_measures[length(eval_measures)+1] <- compute_auc(pred_model, test_df)
    names(eval_measures)[length(eval_measures)] <- 'AUC'
  }
  
  eval_measures

}


## The function extracts some basic evaluation metrics from the model evaluation object
## produced by the confusionMatrix() f. of the caret package
extract_eval_measures <- function(model_eval, metrics) {
  eval_measures <- list()
  by_class_measures <- names(model_eval$byClass)
  overall_measures <- names(model_eval$overall)
  for(m in metrics) {
    if(m %in% by_class_measures)
      eval_measures[[m]] <- as.numeric(model_eval$byClass[m])
    else if(m %in% overall_measures)
      eval_measures[[m]] <- as.numeric(model_eval$overall[m])
  }
  unlist(eval_measures)
}


compute_auc <- function(pred_model, test_df) {
  preds_prob <- predict(pred_model, test_df, type = 'prob')
  require(pROC)
  # the 1st argument of the roc f. is the probablity of the 
  # positive class, which is, by default, the class that 
  # corresponds to the level 1 of the class (factor) variable
  # (N.B. roc f. uses terms 'cases' and 'controls' to refer to
  # the positive and negative classes, respectively)
  roc <- roc(predictor=preds_prob[,2], response=test_df$Label)
  plot.roc(roc, print.auc = TRUE)
  roc$auc
}
