# Performing Optical Character Recognition with Support Vector Machines ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 7:
# Black Box Methods - Neural Networks and Support Vector Machines
#
# The original code is made with {kernlab}. I wanted to see how one could
# recreate it using mainly {tidymodels} and {tidyverse}.
#
# You can find the original code and the dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter07

## 1. Loading libraries (in the order they get used) ----
library(conflicted)
library(tidyverse)
library(tidymodels)
library(janitor)


## 2. Exploring and preparing the data ----

### Read in data and examine structure ----
letters_tbl <- read_csv("support_vector_machines/data/letterdata.csv") %>%
  mutate(across(where(is.character), as.factor))

letters_tbl %>% 
  glimpse()


## 3. Creating the recipe ----

### Apply normalization to entire data frame ----
recipe_obj <- recipe(
  letter ~ .,
  data = letters_tbl
)
recipe_obj

letters_baked_tbl <- recipe_obj %>%
  prep() %>%
  bake(new_data = NULL)
letters_baked_tbl

### Create training and test data ----
letters_split <- initial_time_split(
  letters_baked_tbl,
  prop = 16000 / 20000
)
letters_train <- training(letters_split)
letters_test  <- testing(letters_split)


## 4. Training a model on the data ----

### Create model specification ----
model_spec <- svm_linear(
  engine       = "kernlab",
  mode         = "classification",
  cost         = NULL,
  margin       = NULL
) %>%
  translate()
model_spec

### Fit the model ----
model_fit <- fit(
  model_spec,
  letter ~ .,
  letters_train
)
model_fit

### Make the predictions (you could skip this step) ----
letters_test_pred <- predict(
  model_fit,
  new_data = letters_test
)
letters_test_pred

### Add the predictions to the test tibble ----
letters_test_with_pred <- augment(model_fit, letters_test)
letters_test_with_pred


## 5. Evaluating model performance ----

### Predictions on testing dataset ----
letters_test_with_pred %>%
  tabyl(letter, .pred_class)

### Look only at agreement vs. non-agreement ----
# Construct a vector of TRUE/FALSE indicating correct/incorrect predictions
letters_test_with_pred %>%
  mutate(
    agreement = case_when(
      letter == .pred_class ~ TRUE,
      TRUE                  ~ FALSE
    )
  ) %>%
  tabyl(agreement)


## 6. Improving model performance ----

### Change to a RBF kernel ----
model_spec_rbf <- svm_rbf(
  engine       = "kernlab",
  mode         = "classification",
  cost         = NULL,
  margin       = NULL,
  rbf_sigma    = NULL
) %>%
  translate()
model_spec_rbf

### Fit the model ----
model_fit_rbf <- fit(
  model_spec_rbf,
  letter ~ .,
  letters_train
)
model_fit_rbf

### Make the predictions (you could skip this step) ----
letters_test_pred_rbf <- predict(
  model_fit_rbf,
  new_data = letters_test
)
letters_test_pred_rbf

### Add the predictions to the test tibble ----
letters_test_with_pred_rbf <- augment(model_fit_rbf, letters_test)
letters_test_with_pred_rbf

### Predictions on testing dataset ----
letters_test_with_pred_rbf %>%
  tabyl(letter, .pred_class)

### Look only at agreement vs. non-agreement ----
# Construct a vector of TRUE/FALSE indicating correct/incorrect predictions
letters_test_with_pred_rbf %>%
  mutate(
    agreement = case_when(
      letter == .pred_class ~ TRUE,
      TRUE                  ~ FALSE
    )
  ) %>%
  tabyl(agreement)

### Test various values of the cost parameter ----
cost_values <- c(1, seq(from = 5, to = 40, by = 5))

accuracy_values <- map_dbl(cost_values, function(x) {

  model_spec_rbf <- svm_rbf(
    engine       = "kernlab",
    mode         = "classification",
    cost         = {{ x }},
    margin       = NULL,
    rbf_sigma    = NULL
  ) %>%
    translate()

  model_fit_rbf <- fit(
    model_spec_rbf,
    letter ~ .,
    letters_train
  )

  letters_test_pred_rbf <- predict(
    model_fit_rbf,
    new_data = letters_test
  ) %>% as_vector()

  agree <- ifelse(letters_test_pred_rbf == letters_test$letter, 1, 0)

  accuracy <- sum(agree) / nrow(letters_test)

  return(accuracy)

})

### Bind together the cost parameter and accuracy values ----
cost_vs_accuracy_tbl <- bind_cols(
  cost_values,
  accuracy_values,
) %>%
  rename(
    cost_values     = ...1,
    accuracy_values = ...2
  )

### Visualize to find the optimal cost parameter value ----
cost_vs_accuracy_tbl %>%
  ggplot(aes(cost_values, accuracy_values)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  labs(
    x = "Cost Parameter Value",
    y = "Accuracy"
  )

### Make sure you have the right optimal cost value for the best accuracy ----
cost_vs_accuracy_tbl %>%
  slice_max(accuracy_values)

### Pull the first cost_value that has the max accuracy_value ----
.max_accuracy <- cost_vs_accuracy_tbl %>%
  slice_max(accuracy_values) %>%
  slice(1) %>%
  pull(cost_values)


## 7. Fitting the model with the optimal cost value (that was just pulled) ----

### Give model specification ----
model_spec_best <- svm_rbf(
  engine       = "kernlab",
  mode         = "classification",
  cost         = {{ .max_accuracy }},
  margin       = NULL,
  rbf_sigma    = NULL
) %>%
  translate()

### Fit the model ----
model_fit_best <- fit(
  model_spec_best,
  letter ~ .,
  letters_train
)

### Add the predictions to the test tibble ----
letters_test_with_pred_best <- augment(model_fit_best, letters_test)

### Predictions on testing dataset ----
letters_test_with_pred_best %>%
  tabyl(letter, .pred_class)

### Look only at agreement vs. non-agreement ----
# Construct a vector of TRUE/FALSE indicating correct/incorrect predictions
letters_test_with_pred_best %>%
  mutate(
    agreement = case_when(
      letter == .pred_class ~ TRUE,
      TRUE                  ~ FALSE
    )
  ) %>%
  tabyl(agreement)