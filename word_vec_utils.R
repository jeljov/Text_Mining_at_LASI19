# The function creates a data frame out of the word vectors 
# that originate from a pre-trained GloVe model (m_glove)
get_word_vectors_df <- function(m_glove, verbose = FALSE) {
  
  # initialize space for values and the names of each word in the model
  n_words <- length(m_glove)
  vals <- list()
  names <- character(n_words)
  
  # loop through to gather values and names of each word
  for(i in 1:n_words) {
    if (verbose) {
      if(i %% 5000 == 0) {print(i)}
    }
    this_vec <- m_glove[i]
    this_vec_unlisted <- unlist(strsplit(this_vec, " "))
    this_vec_values <- as.numeric(this_vec_unlisted[-1])  
    this_vec_name <- this_vec_unlisted[1]
    
    vals[[i]] <- this_vec_values
    names[i] <- this_vec_name
  }
  
  # convert the list to a data frame and attach the names
  glove_df <- data.frame(vals)
  names(glove_df) <- names
  
  glove_df
}



# The function creates a subset of the given word vector model 
# (the 'model' parameter), which focuses on the given search term or terms
# (the 'search_terms' parameter). In particular, it extracts 'n_nearest' terms 
# closest to the given search term(s), reduces the dimensionality of word vectors
# using t-SNE, and plots the reduced vectors using ggplot2 and ggrepel. 
# The combination of ggplot2 and ggrepel creates a red point for each term and
# allows the label for each term to be offset, thus improving readability.
# The last parameter (words_out) determines if the function returns a vector 
# of the ploted words. 
# The function is adapted from:
# http://programminghistorian.github.io/ph-submissions/lessons/getting-started-with-word-embeddings-in-r
w2v_plot <- function(model, search_terms, n_nearest, out_dir, words_out = FALSE) {
  
  require(Rtsne)
  require(ggplot2)
  require(ggrepel)
  require(wordVectors)
  
  # Identify the nearest n_nearest words to the average vector of search terms
  neighbours <- closest_to(model, model[[search_terms]], n_nearest, fancy_names = FALSE)
  wordlist <- neighbours$word
  
  # Create a subset vector space model
  new_model <- model[[wordlist, average = F]]
  
  # Run Rtsne to reduce new Word Embedding Model to 2D (Barnes-Hut)
  reduction <- Rtsne(as.matrix(new_model),
                     check_duplicates = F,
                     pca = F, 
                     verbose = TRUE)
  
  # The Y component of the Rtsne result stores, for each term, values 
  # for the terms' two dimensions
  # Extract Y (positions for plot) as a data frame and add row names
  reduced_model <- as.data.frame(reduction$Y)
  rownames(reduced_model) <- rownames(new_model)
  head(reduced_model)
  
  ref_name <- paste(search_terms, collapse = ", ")
  
  # Create a plot of the t-SNE reduced model and save as jpeg
  tsne_plot <- ggplot(reduced_model) +
        geom_point(aes(x = V1, y = V2), color = "red") +
        geom_text_repel(aes(x = V1, y = V2, label = rownames(reduced_model))) +
        xlab("Dimension 1") +
        ylab("Dimension 2 ") +
        # geom_text(fontface = 2, alpha = .8) +
        theme_bw(base_size = 12) +
        theme(legend.position = "none") +
        ggtitle(paste0("2D reduction of Word2Vec Model for word(s): ", ref_name, " using t_SNE"))
  
  ggsave(paste0(ref_name, ".jpeg"), plot = tsne_plot, path = out_dir, width = 24,
         height = 18, dpi = 100)
  
  if(words_out)
    return(wordlist)
}