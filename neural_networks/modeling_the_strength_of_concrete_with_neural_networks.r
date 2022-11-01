# Modeling the Strength of Concrete with Neural Networks ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 7:
# Black Box Methods - Neural Networks and Support Vector Machines
#
# The original code is made with {neuralnet}. I wanted to see how one could
# recreate it using mainly {tidymodels} and {tidyverse}.
#
# You can find the original code and the dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter07

## 1. Loading libraries (in the order they get used) ----
library(tidyverse)
library(tidymodels)
library(corrr)


## 2. Exploring and preparing the data ----

### Read in data and examine structure ----
concrete_tbl <- read_csv("neural_networks/data/concrete.csv")

concrete_tbl %>% glimpse()

### Check the minimum and maximum strength ----
concrete_tbl %>%
  select(strength) %>%
  summary()


## 3. Creating the recipe ----

### Apply normalization to the numeric predictors ----
recipe_obj <- recipe(
  strength ~ .,
  data = concrete_tbl
) %>%
  step_range(
    all_numeric_predictors(),
    min = 0,
    max = 1
  )
recipe_obj

concrete_normalized_tbl <- recipe_obj %>%
  prep() %>%
  bake(new_data = NULL)
concrete_normalized_tbl

### Create training and test data ----
concrete_split <- initial_time_split(
  concrete_normalized_tbl,
  prop = 773 / 1030
)
concrete_train <- training(concrete_split)
concrete_test  <- testing(concrete_split)


## 3. Training a model on the data ----

# nnet is the engine (needs to be installed if not already):
# install.packages("nnet")

# It is used as the engine for {parsnip}'s mlp() function.
# And since we are predicting strength, we choose regression as the mode.

### Create model specification ----
model_spec <- mlp(
  engine       = "nnet",
  mode         = "regression",
  hidden_units = 5,
  penalty      = 0,
  epochs       = 100
) %>%
  translate()
model_spec

### Fit the model ----
model_fit <- fit(
  model_spec,
  strength ~ .,
  concrete_train
)
model_fit

### Take a closer look at the model ----
summary(model_fit$fit)

### Make the predictions (you could skip this step) ----
concrete_test_pred <- predict(
  model_fit,
  new_data = concrete_test,
  type = "numeric"
)
concrete_test_pred

### Add the predictions to the test tibble ----
concrete_test_with_pred <- augment(model_fit, concrete_test)
concrete_test_with_pred

### Metrics ----
concrete_test_with_pred %>% metrics(strength, .pred)

### Visualize the network topology ----
# I have yet to find a working solution to create this visualization.
# If you know how to do it, please let me know!


## 4. Evaluating model performance ----

### Examine the correlation between predicted and actual values ----
concrete_test_with_pred %>%
  select(.pred, strength) %>%
  correlate()


## 5. Improving model performance with two hidden layers and custom activation function ----

### Create model specification ----
model_spec_2 <- mlp(
  engine       = "nnet",
  mode         = "regression",
  hidden_units = 5,
  penalty      = 0.1,
  epochs       = 100,
) %>%
  translate()
model_spec_2

### Fit the model ----
model_fit_2 <- fit(
  model_spec_2,
  strength ~ .,
  concrete_train
)
model_fit_2

### Take a closer look at the model ----
summary(model_fit_2$fit)

### Make the predictions (you could skip this step) ----
concrete_test_pred_2 <- predict(
  model_fit_2,
  new_data = concrete_test,
  type = "numeric"
)
concrete_test_pred_2

### Add the predictions to the test tibble ----
concrete_test_with_pred_2 <- augment(model_fit_2, concrete_test)
concrete_test_with_pred_2

### Metrics ----
concrete_test_with_pred_2 %>% metrics(strength, .pred)

### Examine the correlation between predicted and actual values ----
concrete_test_with_pred_2 %>%
  select(.pred, strength) %>%
  correlate()
