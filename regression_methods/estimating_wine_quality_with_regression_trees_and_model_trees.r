# Estimating Wine Quality with Regression Trees and Model Trees ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 6:
# Forecasting Numeric Data - Regression Methods.
#
# The original code is made with {rpart}, {rpart.plot} and {Cubist}. I wanted to
# see how one could recreate it using mainly {tidymodels}, {tidyverse} and {rules}.
#
# You can find the original code and the dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter06

## 1. Loading libraries (in the order they get used) ----
library(conflicted)
  conflict_prefer("filter", "dplyr", "stats")
library(tidyverse)
library(tidymodels)
library(rattle)
library(corrr)
library(rules)


## 2. Preparing and exploring the data ----
wine_tbl <- read_csv("regression_methods/data/whitewines.csv") %>%
  rename_all(
    ~str_replace_all(., "\\s+", "_") %>%
      tolower()
  )
wine_tbl

glimpse(wine_tbl)

### The distribution of quality ratings ----
wine_tbl %>%
  ggplot(aes(quality)) +
  geom_histogram()

### Summary statistics of the wine data ----
summary(wine_tbl)

wine_split <- initial_time_split(
  wine_tbl,
  prop = 3750 / 4898
)
wine_train <- training(wine_split)
wine_test  <- testing(wine_split)


## 3. Training a model on the data ----
# regression tree using rpart
model_spec <- decision_tree(
  mode            = "regression",
  engine          = "rpart",
  cost_complexity = NULL,
  tree_depth      = NULL,
  min_n           = NULL
) %>%
  translate()
model_spec

### Fit the model ----
model_fit <- fit(
  model_spec,
  quality ~ .,
  wine_train
)

### Get basic information about the tree ----
model_fit

### Get more detailed information about the tree ----
summary(model_fit$fit)

### Adjust plot margins to make the visualization work better
par(mar = c(1, 1, 1, 1))

### Use the rattle package to create a visualization ----
model_fit$fit %>%
  fancyRpartPlot(cex = 0.5)


## 4. Evaluate model performance ----

### Generate predictions for the testing dataset (you could skip this step) ----
wine_test_pred <- predict(
  object   = model_fit,
  new_data = wine_test,
  type     = "numeric"
)
wine_test_pred

### Add the predictions to the test tibble ----
wine_test_with_pred_tbl <- augment(model_fit, wine_test)
wine_test_with_pred_tbl

### Compare the distribution of actual values vs. predicted values ----
wine_test_with_pred_tbl %>%
  select(quality, .pred) %>%
  summary()

### Compare the correlation ----
wine_test_with_pred_tbl %>%
  select(quality, .pred) %>%
  correlate()

### Mean absolute error between actual and predicted values ----
wine_test_with_pred_tbl %>%
  metrics(quality, .pred) %>%
  filter(.metric == "mae")

### Mean absolute error between actual values and mean value ----
mean_value <- wine_train$quality %>%
  mean()
mean_value

mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

mae(wine_test$quality, mean_value)


## 5. Improving model performance ----
# train a Cubist model tree
model_spec_cubist <- cubist_rules(
  mode            = "regression",
  engine          = "Cubist",
  committees      = NULL,
  neighbors       = NULL,
  max_rules       = NULL
) %>%
  translate()
model_spec_cubist

### Fit the model ----
model_fit_cubist <- fit(
  model_spec_cubist,
  quality ~ .,
  wine_train
)

### Display basic information about the model tree ----
model_fit_cubist

### Display the tree itself ----
summary(model_fit_cubist$fit)

### Generate predictions for the model ----
wine_test_pred_cubist <- predict(
  object   = model_fit_cubist,
  new_data = wine_test,
  type     = "numeric"
)
wine_test_pred_cubist

### Summary statistics about the predictions ----
wine_test_pred_cubist %>%
  summary()

### Add the predictions to the test tibble ----
wine_test_with_pred_cubist <- augment(model_fit_cubist, wine_test)
wine_test_with_pred_cubist

### Compare the distribution of actual values vs. predicted values ----
wine_test_with_pred_cubist %>%
  select(quality, .pred) %>%
  summary()

### Correlation between the true and predicted values ----
wine_test_with_pred_cubist %>%
  select(quality, .pred) %>%
  correlate()

### Mean absolute error of true and predicted values ----
wine_test_with_pred_cubist %>%
  metrics(quality, .pred) %>%
  filter(.metric == "mae")


## 6. Creating a function to test the model(s) with another dataset ----

# The assumption here is that you have already taken step 1.

# Preparing and exploring the other wine dataset
red_wine_tbl <- read_csv("regression_methods/data/redwines.csv") %>%
  rename_all(
    ~str_replace_all(., "\\s+", "_") %>%
      tolower()
  )
red_wine_tbl

### The distribution of quality ratings ----
red_wine_tbl %>%
  ggplot(aes(quality)) +
  geom_histogram()

### Summary statistics of the wine data ----
summary(red_wine_tbl)

### Create the function

predict_wine_quality <- function(
  .engine    = c("rpart", "Cubist"),
  .winecolor = c("red", "white")
) {

  # Check that the wine color is valid
  if (!.winecolor %in% c("red", "white")) stop("Choose a wine color: red or white")

  # Write out the path so that you can insert the wine color in there
  path <- str_glue("regression_methods/data/{.winecolor}wines.csv")

  # Read in the data
  wine_tbl <- read_csv(path) %>%
    rename_all(
      ~str_replace_all(., "\\s+", "_") %>%
        tolower()
    )

  # Make the train/test split. It's not randomizing, as the data already is
  wine_split <- initial_time_split(
    wine_tbl,
    prop = 0.75
  )
  wine_train <- training(wine_split)
  wine_test  <- testing(wine_split)

  # Create the model based on the engine chosen
  if (.engine == "rpart") {

    model_spec <- decision_tree(
      mode            = "regression",
      engine          = "rpart"
    ) %>%
      translate()

  } else if (.engine == "Cubist") {

    model_spec <- cubist_rules(
      mode            = "regression",
      engine          = "Cubist"
    ) %>%
      translate()

  } else {

    stop("Choose an engine: rpart (decision tree) or Cubist (rules)")

  }

  # Fit the model
  model_fit <- fit(
    model_spec,
    quality ~ .,
    wine_train
  )

  # Add the predictions
  wine_test_with_pred_tbl <- augment(model_fit, wine_test)

  # Get the metrics
  wine_test_with_pred_tbl %>%
    metrics(quality, .pred)

}

### Test the function ----
predict_wine_quality(
  .engine    = "Cubist",
  .winecolor = "white"
)
