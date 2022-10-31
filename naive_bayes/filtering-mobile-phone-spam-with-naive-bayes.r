# Filtering Mobile Phone Spam with the Naive Bayes Algorithm ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 4:
# Probabilistic Learning - Classification Using Naive Bayes.
#
# The original code is made with a lot of base R, {e1071} and {gmodels}. I
# wanted to see how one could recreate it using mainly {textrecipes},
# {tidymodels}, {tidytext} and {tidyverse}.
#
# You can find the original code and the slightly modified dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter04

## 1. Loading libraries (in the order they get used) ----
library(tidyverse)
library(tidytext)
library(SnowballC)   # for stemming
library(wordcloud2)
library(textrecipes)
library(tidymodels)
library(discrim)     # for naive_Bayes()


## 2. Exploring and preparing the data ----

### Read the sms data into the sms tibble, convert spam/ham to factor ----
sms_tbl <- read_csv(
  "naive_bayes/data/sms_spam.csv",
  col_types = "fc"
) %>%
  select(.type = type, everything())

### Examine the structure of the sms data ----
glimpse(sms_tbl)

### Examine the distribution of spam/ham ----
sms_tbl %>%
  count(.type) %>%
  mutate(pct = (n / sum(n) * 100))

### Build a corpus using the {tidytext} package instead of {tm} ----

# This part inspired by a blog post by Julia Silge:
# https://www.tidyverse.org/blog/2020/11/tidymodels-sparse-support/

### Add a row number ----
sms_row_numbers_tbl <- sms_tbl %>%
  mutate(line = row_number())
sms_row_numbers_tbl

### Manual preprocessing (just to see what it's all about) ----
tidy_sms_tbl <- sms_row_numbers_tbl %>%
  unnest_tokens(word, text) %>%
  count(line, word) %>%
  bind_tf_idf(word, line, n)
tidy_sms_tbl

wide_sms_tbl <- tidy_sms_tbl %>%
  dplyr::select(line, word, tf_idf) %>%
  pivot_wider(
    names_from   = word,
    names_prefix = "word_",
    values_from  = tf_idf,
    values_fill  = 0
  )
wide_sms_tbl

### Transform the results to a sparse matrix ----
sparse_sms <- tidy_sms_tbl %>%
  cast_dfm(line, word, tf_idf)
sparse_sms

### Compare the difference in memory usage between wide and sparse formats ----
lobstr::obj_sizes(wide_sms_tbl, sparse_sms)


## 3. Visualizing text data - word clouds ----

# This part inspired by Julia Silge & David Robinson's book Text Mining with R:
# A Tidy Approach: https://www.tidytextmining.com/

### Count word frequencies ----
frequency_tbl <- sms_row_numbers_tbl %>%

  # One word per one row
  unnest_tokens(word, text) %>%

  # Stemming
  mutate(word = wordStem(word)) %>%

  # Count the words
  count(.type, word) %>%

  # Count the proportion of words
  with_groups(
    .type,
    mutate,
    proportion = n / sum(n)
  ) %>%

  # Reorder the columns
  dplyr::select(-n) %>%
  pivot_wider(names_from = .type, values_from = proportion) %>%
  pivot_longer(
    cols      = c("ham", "spam"),
    names_to  = ".type",
    values_to = "freq"
  )

### Subset the frequency data into two groups, spam and ham ----
spam_tbl <- frequency_tbl %>%
  filter(.type == "spam") %>%
  dplyr::select(-.type) %>%
  drop_na()

ham_tbl <- frequency_tbl %>%
  filter(.type == "ham") %>%
  dplyr::select(-.type) %>%
  drop_na()

### Word cloud ----

# This part inspired by a blog post by Céline Van den Rul:
# https://towardsdatascience.com/create-a-word-cloud-with-r-bde3e7422e8a

# One for ham...
wordcloud2(
  data  = ham_tbl,
  size  = 2,
  color = "random-dark"
)

# ...and another for spam
wordcloud2(
  data  = spam_tbl,
  size  = 2,
  color = "random-dark"
)


## 4. Creating the recipe and splitting the data ----

### Create the recipe ----
text_recipe_obj <- recipe(
  .type ~ text,
  data = sms_row_numbers_tbl
) %>%
  step_tokenize(text)  %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text)
text_recipe_obj

# Bake it
sms_baked_tbl <- text_recipe_obj %>%
  prep() %>%
  bake(new_data = NULL)

# Simplify the tf-idf to yes/no
sms_baked_longer_tbl <- sms_baked_tbl %>%
  mutate(across(where(is.numeric),
                ~ case_when(
                  . > 0 ~ "Yes",
                  TRUE   ~ "No"
                ))) %>%
  # Rename the columns back to words
  rename_with(
    ~ tolower(gsub("tfidf_text_", "", .x)),
    .cols = starts_with("tfidf_text_")
  )
sms_baked_longer_tbl

### Create training and test data ----
# Not randomly, because the messages weren't in any particular order
sms_split <- initial_time_split(
  sms_baked_longer_tbl,
  prop = 0.75
)
sms_train <- training(sms_split)
sms_test  <- testing(sms_split)


## 5. Training a model on the data ----

# naivebayes is the engine (needs to be installed if not already):
# install.packages("discrim") AND
# install.packages("naivebayes")

# It is used as the engine for {parsnip}'s naive_Bayes() function. And since
# we are classifying, that is the mode we choose.

# The simple reason is I couldn't get klaR (the other engine) to work. If you
# know how, please comment on GitHub. It would be great to get to test what the
# difference between the two engines are.

### Model specification ----
model_spec <- naive_Bayes(
  engine     = "naivebayes",
  mode       = "classification",
  smoothness = NULL,
  Laplace    = NULL
) %>%
  translate()
model_spec

### Fit the model ----
model_fit <- fit(
  model_spec,
  .type ~ .,
  sms_train
)
model_fit

### Make the predictions (you could skip this step) ----
sms_test_pred <- predict(
  object   = model_fit,
  new_data = sms_test,
  type     = "class"
)
sms_test_pred

### Add the predictions to the test tibble ----
sms_test_with_pred_tbl <- augment(model_fit, sms_test)
sms_test_with_pred_tbl


## 6. Evaluating model performance ----

### Create a confusion matrix ----
conf_mat <- conf_mat(
  data     = sms_test_with_pred_tbl,
  truth    = .type,
  estimate = .pred_class
)
conf_mat

### Visualize the confusion matrix ----
conf_mat %>% autoplot(type = "heatmap")
conf_mat %>% autoplot(type = "mosaic")

### Visualize the ROC curve ----
sms_test_with_pred_tbl %>%
  roc_curve(
    truth    = .type,
    estimate = .pred_ham
  ) %>%
  autoplot()

### Calculate the ROC AUC (area under the curve) ----
sms_roc_auc <- sms_test_with_pred_tbl %>%
  roc_auc(
    truth    = .type,
    estimate = .pred_ham
  )
sms_roc_auc

### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...
classification_metrics <- conf_mat(
  sms_test_with_pred_tbl,
  truth    = .type,
  estimate = .pred_class
) %>%
  summary()
classification_metrics


## 7. Improving model performance ----

# Basically, the same as before, but with Laplace = 1

### Model specification ----
model_spec <- naive_Bayes(
  engine     = "naivebayes",
  mode       = "classification",
  smoothness = NULL,
  Laplace    = 1
) %>%
  translate()
model_spec

### Fit the model ----
model_fit <- fit(
  model_spec,
  .type ~ .,
  sms_train
)
model_fit

### Make the predictions (you could skip this step) ----
sms_test_pred <- predict(
  object   = model_fit,
  new_data = sms_test,
  type     = "class"
)
sms_test_pred

### Add the predictions to the test tibble ----
sms_test_with_pred_tbl <- augment(model_fit, sms_test)
sms_test_with_pred_tbl


## 6. Evaluating model performance ----

### Create a confusion matrix ----
conf_mat <- conf_mat(
  data     = sms_test_with_pred_tbl,
  truth    = .type,
  estimate = .pred_class
)
conf_mat

### Visualize the confusion matrix ----
conf_mat %>% autoplot(type = "heatmap")
conf_mat %>% autoplot(type = "mosaic")

### Visualize the ROC curve ----
sms_test_with_pred_tbl %>%
  roc_curve(
    truth    = .type,
    estimate = .pred_ham
  ) %>%
  autoplot()

### Calculate the ROC AUC (area under the curve) ----
sms_roc_auc <- sms_test_with_pred_tbl %>%
  roc_auc(
    truth    = .type,
    estimate = .pred_ham
  )
sms_roc_auc

### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...
classification_metrics <- conf_mat(
  sms_test_with_pred_tbl,
  truth    = .type,
  estimate = .pred_class
) %>%
  summary()
classification_metrics


## 8. Creating a function to help evaluate the model further ----

# The assumption here is that you have already gone through steps 1. to 4.
# What we're potentially tuning here are the arguments .smoothness and .Laplace
# Check out the book and/or the documentation for further info about them!

classify_with_naive_bayes <- function(
  .smoothness  = NULL,
  .laplace     = NULL
) {

  # Model specification
  model_spec <- naive_Bayes(
    engine     = "naivebayes",
    mode       = "classification",
    smoothness = .smoothness,
    Laplace    = .laplace
  ) %>%
    translate()
  model_spec

  # Fit the model
  model_fit <- fit(
    model_spec,
    .type ~ .,
    sms_train
  )
  model_fit

  # Add the predictions to the test tibble
  sms_test_with_pred_tbl <- augment(model_fit, sms_test)
  sms_test_with_pred_tbl

  # Create a confusion matrix
  conf_mat <- conf_mat(
    data     = sms_test_with_pred_tbl,
    truth    = .type,
    estimate = .pred_class
  )

  # Print the confusion matrix
  conf_mat %>% autoplot(type = "heatmap")

}

### Test the function ----
classify_with_naive_bayes(
  .smoothness  = 1,
  .laplace     = 1
)
