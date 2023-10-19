# Hyperparameter Optimization with tidymodels ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 11:
# Improving Model Performance
#
# The original code is made with {caret}. I wanted to see how one could recreate
# it using mainly {tidymodels} and {tidyverse}.
#
# You can find the original code and the slightly modified dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter11

## 1. Loading libraries (in the order they get used) ----
library(conflicted)
library(tidyverse)
library(tidymodels)


## 2. Loading the dataset ----
credit_tbl <- read_csv("decision_trees_and_rules/data/credit.csv")


## 3. Creating the recipe and splitting the data ----

### Convert strings to factors ----
recipe_obj <- recipe(
    default ~ .,
    data = credit_tbl
) %>%
    step_string2factor(all_nominal())
recipe_obj

credit_factorized_tbl <- recipe_obj %>%
    prep() %>%
    bake(new_data = NULL)
credit_factorized_tbl

### Create training and test data (randomly) ----
set.seed(123)
credit_split <- initial_split(
    credit_factorized_tbl,
    prop = 0.9,
    strata = default
)

credit_train <- training(credit_split)
credit_test  <- testing(credit_split)

### Check the proportion of class variable ----
credit_train %>%
    count(default) %>%
    mutate(pct = (n / sum(n) * 100))

credit_test %>%
    count(default) %>%
    mutate(pct = (n / sum(n) * 100))


## 4. Creating a simple model ----

### Create specifications with the default tuning parameters for rpart ----
# tree_depth = 30, min_n = 2 and cost_complexity = 0.01

set.seed(123)
tune_spec_simple <- decision_tree(
    mode            = "classification",
    engine          = "rpart"
)
tune_spec_simple

### Fit the model ----
model_fit_simple <- fit(
    tune_spec_simple,
    default ~ .,
    credit_train
)

model_fit_simple %>% 
    extract_fit_engine() %>%
    summary()

### Make the predictions ----
credit_simple <- augment(model_fit_simple, credit_test)
credit_simple

### Create a confusion matrix ----
conf_mat_simple <- conf_mat(
    data     = credit_simple,
    truth    = default,
    estimate = .pred_class
)
conf_mat_simple

### Visualize the ROC curve ----
credit_simple %>%
    roc_curve(
        truth    = default,
        estimate = .pred_no
    ) %>%
    autoplot()

### Calculate the ROC AUC (area under the curve) ----
credit_roc_auc_simple <- credit_simple %>%
    roc_auc(
        truth    = default,
        estimate = .pred_no
    )
credit_roc_auc_simple

### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...

classification_metrics_simple <- conf_mat(
    credit_simple,
    truth    = default,
    estimate = .pred_class
) %>%
    summary()
classification_metrics_simple


## 5. Generating a tuning grid ----

### Create a specification with tuning placeholders ----
set.seed(123)
tune_spec <- decision_tree(
    mode            = "classification",
    engine          = "rpart",
    tree_depth      = tune(), 
    min_n           = tune(),
    cost_complexity = tune()
)
tune_spec

### Create a regular grid ----
tree_grid <- grid_random(
    extract_parameter_set_dials(tune_spec),
    levels = 4
)
tree_grid


## 6. Tuning along the grid ----

### Create CV folds of the credit_train ----
folds <- vfold_cv(
    credit_train,
    v = 10
)
folds

### Tune along the grid ----
tune_results <- tune_grid(
    tune_spec, 
    default ~ .,
    resamples = folds,
    grid      = tree_grid,
    metrics   = metric_set(accuracy)
)

### Plot the tuning results ----
autoplot(tune_results)


## 7. Picking the winner ----

### Select the parameters that perform best ----
final_params <- select_best(tune_results)
final_params

### Finalize the specification ----
best_spec <- finalize_model(tune_spec, final_params)
best_spec

### Build the final model ----
final_model <- fit(
    best_spec,
    default ~ .,
    credit_train
)
final_model

final_model %>%
    extract_fit_engine() %>%
    summary()

### Add the predictions to the test tibble ----
credit_with_pred_tbl <- augment(final_model, credit_test)
credit_with_pred_tbl


## 8. Evaluating model performance ----

### Create a confusion matrix ----
conf_mat <- conf_mat(
    data     = credit_with_pred_tbl,
    truth    = default,
    estimate = .pred_class
)
conf_mat

### Visualize the ROC curve ----
credit_with_pred_tbl %>%
    roc_curve(
        truth    = default,
        estimate = .pred_no
    ) %>%
    autoplot()

### Calculate the ROC AUC (area under the curve) ----
credit_roc_auc <- credit_with_pred_tbl %>%
    roc_auc(
        truth    = default,
        estimate = .pred_no
    )
credit_roc_auc

### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...
classification_metrics <- conf_mat(
    credit_with_pred_tbl,
    truth    = default,
    estimate = .pred_class
) %>%
    summary()
classification_metrics
