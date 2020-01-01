## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for LASI'19 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi19/lasi19-workshops-tutorials/)


## ON THE 20 NEWSGROUPS DATASET
## 
## The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup 
## documents (forum posts), partitioned (nearly) evenly across 20 different newsgroups,
## each corresponding to a different topic.
##
## In case the term "newsgroup" is new to you: 
## A newsgroup is an online discussion forum accessible through Usenet.
## Usenet is a decentralized computer network, like Internet, initially primarily used by 
## students and staff in universities across the U.S. to communicate by sharing messages, 
## news, and updates. It is still in active use with over 100K active newsgroups 
## (see: https://www.binsearch.info/groupinfo.php)
##
## The dataset is publicly available from: http://qwone.com/~jason/20Newsgroups/ 
## Note that the page provides 3 versions of the dataset; the one that should be 
## downloaded is "20news-bydate.tar.gz" (here is a direct link: 
## http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)

##############################################
## SET UP the SESSION: 
## - INSTALL and LOAD the REQUIRED LIBRARIES
## - LOAD the AUXILIARY SCRIPTS
##############################################

# The script makes use of the following R packages:
# - dplyr, tidyr, purrr, reader - for general data analysis tasks 
# - stringr - for advanced string processing
# If you miss any of these packages, install them, before proceeding with the script
# install.packages(c("<package_name_1>", "<package_name_2>", ...))

# Load the required libraries
library(dplyr) # v. 0.8.3
library(tidyr) # v. 1.0.0.0
library(purrr) # v. 0.3.2
library(readr) # v. 1.3.1
library(stringr) # v. 1.4.0

# Load an R script with a set of auxiliary functions
source("tm_utils.R")

###############################
## READ TEXT FROM SOURCE FILES
###############################

# We’ll start by reading in posts from the "20news-bydate-train" and 
# "20news-bydate-test" folders, which are organized in sub-folders, 
# each corresponding to one newsgroup, with one file per post

training_folder <- "data/20news/20news-bydate-train"
test_folder <- "data/20news/20news-bydate-test"

# Read in the contents of all posts from the train dataset
# Note: read_folder() is a utility function defined in the tm_utils.R 
raw_train_data <- tibble(folder = dir(training_folder, full.names = TRUE)) %>%
  mutate(folder_content = map(folder, read_folder)) %>%  # each mapping iteration results in a df; 
  unnest(cols = one_of('folder_content')) %>%       # unnest 'composes' individual dfs into a large df
  transmute(newsgroup = basename(folder), id, content)

# Do the same for the test dataset
raw_test_data <- tibble(folder = dir(test_folder, full.names = TRUE)) %>%
  mutate(folder_content = map(folder, read_folder)) %>%
  unnest(cols = one_of('folder_content')) %>%       
  transmute(newsgroup = basename(folder), id, content)

# Examine the newsgroups that are included in the training and test datasets, 
# and the number of posts in each one
raw_train_data %>%
  group_by(newsgroup) %>%
  summarise(post_count = n_distinct(id)) %>%
  arrange(post_count)
raw_test_data %>%
  group_by(newsgroup) %>%
  summarise(post_count = n_distinct(id)) %>%
  arrange(post_count)

################################
## PRE-PROCESS (CLEAN) THE TEXT 
################################

# Each post has some extra text that is not representative of the communication between 
# newsgroup members but constitutes the metadata of the exchanged posts.
# For example, every post has a header, containing fields such as 'from:' or 'in_reply_to:' 
# Some also have automated email signatures, which occur after a line containing just dashes 
# (e.g. '--' or '---')
# As an example, examine post 54156
raw_train_data %>% filter(id==54156) %>% pull(content)

# Remove the post header and the automated signature
# Note: the remove_header_and_signature() f. is defined in the tm_utils.R
cleaned_train_data <- remove_header_and_signature(raw_train_data)
cleaned_test_data <- remove_header_and_signature(raw_test_data)

# Check the example post after the first cleaning step
cleaned_train_data %>% filter(id==54156) %>% pull(content)

# Many lines also have nested text representing quotes from other users, 
# typically starting with a line like:
# (1) "In article <snelson3.8.0@uwsuper.edu> snelson3@uwsuper.edu (SCOTT R. NELSON) writes:" or
# (2) ">The rotation has changed due to..." or
# (3) "}first I thought it was an 'RC31'.."
# Remove lines such as (1), and remove the "quotation" characters (>, })
# from lines such as 2) and 3)
# (clean_quoted_text() is defined in the tm_utils.R script)
cleaned_train_data <- clean_quoted_text(cleaned_train_data)
cleaned_test_data <- clean_quoted_text(cleaned_test_data)

# Check the example post after the 2nd cleaning step
cleaned_train_data %>% filter(id==54156) %>% pull(content)
cleaned_train_data %>% filter(id==54170) %>% pull(content)

# As the 2nd example implies, in some cases, the previous 
# (2nd) cleaning step should be repeated:
cleaned_train_data <- clean_quoted_text(cleaned_train_data)
cleaned_test_data <- clean_quoted_text(cleaned_test_data)

# The example post after repeating the 2nd cleaning step
cleaned_train_data %>% filter(id==54170) %>% pull(content)

# We will also remove 3 posts from the training set (9704, 9985, and 14991)
# that contain a large amount of strange, non-text content
cleaned_train_data %>% 
  filter(id==9704) %>%
  pull(content)
cleaned_train_data <- cleaned_train_data %>% 
  filter(!(id %in% c(9704, 9985, 14991)))

# Now, merge all lines of text that belong to the same post
# (the merge_post_text() f. is defined in tm_utils.R)
cleaned_train_posts <- merge_post_text(cleaned_train_data)
glimpse(cleaned_train_posts)
cleaned_test_posts <- merge_post_text(cleaned_test_data)
glimpse(cleaned_test_posts)

# Store the cleaned train and test data into .csv files so that 
# they can be used later more easily
cleaned_train_posts %>%
  write_csv("data/20news/20news-bydate-train.csv")
cleaned_test_posts %>%
  write_csv("data/20news/20news-bydate-test.csv")
