# Estimating Wine Quality with Linear Regression ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 6:
# Forecasting Numeric Data - Regression Methods.
#
# The original code is made with {rpart}, {rpart.plot} and {Cubist}. I wanted to
# see how one could recreate it using mainly {tidymodels}, {tidyverse} and {rules}.
#
# You can find the original code and the dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter06

## 1. Loading libraries ----
library(tidyverse)
library(tidymodels)
library(rattle)
library(corrr)
library(rules)


## Step 2: Preparing and exploring the data ----
wine_tbl <- read_csv("linear_regression/data/whitewines.csv") %>% 
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
    prop = 3750/4898
)
wine_train <- training(wine_split)
wine_test  <- testing(wine_split)


## Step 3: Training a model on the data ----
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

### Use the rattle package to create a visualization ----
model_fit$fit %>%
    fancyRpartPlot(cex = 0.55)


## Step 4: Evaluate model performance ----

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

### Compare the distribution of predicted values vs. actual values ----
wine_test_with_pred_tbl %>%
    select(.pred, quality) %>% 
    summary()

### Compare the correlation ----
wine_test_with_pred_tbl %>%
    select(.pred, quality) %>%
    correlate()

### Function to calculate the mean absolute error ----
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}

### Mean absolute error between predicted and actual values ----
MAE(wine_test$quality, wine_test_with_pred_tbl$.pred)

### Mean absolute error between actual values and mean value ----
wine_train$quality %>% 
    mean()

MAE(wine_test$quality, 5.87)


## Step 5: Improving model performance ----
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

### Compare the distribution of predicted values vs. actual values ----
wine_test_with_pred_cubist %>%
    select(.pred, quality) %>% 
    summary()

### Correlation between the predicted and true values ----
wine_test_with_pred_cubist %>%
    select(.pred, quality) %>%
    correlate()

### Mean absolute error of predicted and true values ----
# (uses the custom function defined above)
MAE(wine_test$quality, wine_test_with_pred_cubist$.pred)