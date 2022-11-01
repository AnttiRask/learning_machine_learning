# Identifying Risky Bank Loans Using C5.0 Decision Trees ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 5:
# Divide and Conquer - Classification Using Decision Trees and Rules.
#
# The original code is made with {C50}, {gmodels}, {OneR} and {RWeka}. I
# wanted to see how one could recreate it using mainly {tidymodels} and
# {tidyverse}.
#
# You can find the original code and the slightly modified dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter05

## 1. Loading libraries (in the order they get used) ----
library(conflicted)
library(tidyverse)
library(tidymodels)

## 2. Exploring and preparing the data ----
credit_tbl <- read_csv("decision_trees_and_rules/data/credit.csv")

### Examine the structure of the credit data ----
glimpse(credit_tbl)

### look at two characteristics of the applicant ----
credit_tbl %>%
    count(checking_balance) %>%
    mutate(pct = (n / sum(n) * 100))

credit_tbl %>%
    count(savings_balance) %>%
    mutate(pct = (n / sum(n) * 100))

### look at two characteristics of the loan ----
credit_tbl %>%
    select(months_loan_duration, amount) %>%
    summary()

### look at the class variable ----
credit_tbl %>%
    count(default) %>%
    mutate(pct = (n / sum(n) * 100))


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

# Use set.seed to use the same random number sequence as the original
RNGversion("3.5.2")
set.seed(123)

credit_split <- initial_split(
    credit_factorized_tbl,
    prop = 0.9
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


## 4. Training a model on the data ----

### Model specification ----
model_spec <- decision_tree(
    mode            = "classification",
    engine          = "C5.0",
    cost_complexity = NULL,
    tree_depth      = NULL,
    min_n           = NULL
) %>%
    translate()
model_spec

### Fit the model ----
model_fit <- fit(
    model_spec,
    default ~ .,
    credit_train
)
model_fit

model_fit %>%
    extract_fit_engine() %>%
    summary()

### Make the predictions (you could skip this step) ----
credit_test_pred <- predict(
    object   = model_fit,
    new_data = credit_test,
    type     = "class"
)
credit_test_pred

### Add the predictions to the test tibble ----
credit_test_with_pred_tbl <- augment(model_fit, credit_test)
credit_test_with_pred_tbl


## 5. Evaluating model performance ----

### Create a confusion matrix ----
conf_mat <- conf_mat(
    data     = credit_test_with_pred_tbl,
    truth    = default,
    estimate = .pred_class
)
conf_mat

### Visualize the confusion matrix ----
conf_mat %>% autoplot(type = "heatmap")
conf_mat %>% autoplot(type = "mosaic")

### Visualize the ROC curve ----
credit_test_with_pred_tbl %>%
    roc_curve(
        truth    = default,
        estimate = .pred_no
    ) %>%
    autoplot()

### Calculate the ROC AUC (area under the curve) ----
credit_roc_auc <- credit_test_with_pred_tbl %>%
    roc_auc(
        truth    = default,
        estimate = .pred_no
    )
credit_roc_auc

### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...
classification_metrics <- conf_mat(
    credit_test_with_pred_tbl,
    truth    = default,
    estimate = .pred_class
) %>%
    summary()
classification_metrics


## 6. Improving model performance ----

### Boost the decision tree with 10 trials ----
model_spec_boost_tree <- boost_tree(
    mode            = "classification",
    engine          = "C5.0",
    trees           = 10,
    min_n           = NULL,
    sample_size     = NULL
) %>%
    translate()
model_spec_boost_tree

### Fit the model ----
model_fit_boost_tree <- fit(
    model_spec_boost_tree,
    default ~ .,
    credit_train
)
model_fit_boost_tree

model_fit_boost_tree %>%
    extract_fit_engine() %>%
    summary()

### Make the predictions (you could skip this step) ----
credit_test_pred_boost_tree <- predict(
    object   = model_fit_boost_tree,
    new_data = credit_test,
    type     = "class"
)
credit_test_pred_boost_tree

### Add the predictions to the test tibble ----
credit_test_with_pred_boost_tree <- augment(model_fit_boost_tree, credit_test)
credit_test_with_pred_boost_tree

### Create a confusion matrix ----
conf_mat_boost_tree <- conf_mat(
    data     = credit_test_with_pred_boost_tree,
    truth    = default,
    estimate = .pred_class
)
conf_mat_boost_tree

### Visualize the confusion matrix ----
conf_mat_boost_tree %>% autoplot(type = "heatmap")
conf_mat_boost_tree %>% autoplot(type = "mosaic")

### Visualize the ROC curve ----
credit_test_with_pred_boost_tree %>%
    roc_curve(
        truth    = default,
        estimate = .pred_no
    ) %>%
    autoplot()

### Calculate the ROC AUC (area under the curve) ----
credit_roc_auc_boost_tree <- credit_test_with_pred_boost_tree %>%
    roc_auc(
        truth    = default,
        estimate = .pred_no
    )
credit_roc_auc_boost_tree

### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...
classification_metrics_boost_tree <- conf_mat(
    credit_test_with_pred_boost_tree,
    truth    = default,
    estimate = .pred_class
) %>%
    summary()
classification_metrics_boost_tree


## 6. Creating a function to help evaluate the model further ----

# The assumption here is that you have already gone through steps 1. to 2.
# What we're potentially tuning here are the arguments .tree_depth and .min_n
# for decision_tree, and .trees and .min_n for boost_tree.

classify_with_c5_trees <- function(
    .model           = c("decision_tree", "boost_tree"),
    .mode            = "classification",
    .engine          = "C5.0",
    .tree_depth      = NULL, # for decision_tree
    .trees           = NULL, # for boost_tree
    .min_n           = 1     # for both
) {
    
    # Create the recipe
    recipe_obj <- recipe(
        default ~ .,
        data = credit_tbl
    ) %>%
        step_string2factor(all_nominal())
    
    credit_factorized_tbl <- recipe_obj %>%
        prep() %>%
        bake(new_data = NULL)
    
    # Create training and test data (randomly)
    RNGversion("3.5.2")
    set.seed(123)
    
    credit_split <- initial_split(
        credit_factorized_tbl,
        prop = 0.9
    )
    credit_train <- training(credit_split)
    credit_test  <- testing(credit_split)
    
    # Model specification
    model <- .model
    
    if (model == "decision_tree") {
        
        model_spec <- decision_tree(
            mode            = .mode,
            engine          = .engine,
            tree_depth      = .tree_depth,
            min_n           = .min_n
        ) %>%
            translate()
        
    } else if (model == "boost_tree") {
        
        model_spec <- boost_tree(
            mode            = .mode,
            engine          = .engine,
            trees           = .trees,
            min_n           = .min_n
        ) %>%
            translate()
        
    } else {
        
        stop("The model needs to be either decision_tree or boost_tree!")
        
    }
    
    # Fit the model
    model_fit <- fit(
        model_spec,
        default ~ .,
        credit_train
    )
    
    # Add the predictions to the test tibble
    credit_test_with_pred_tbl <- augment(model_fit, credit_test)
    credit_test_with_pred_tbl
    
    # Create a confusion matrix
    conf_mat <- conf_mat(
        data     = credit_test_with_pred_tbl,
        truth    = default,
        estimate = .pred_class
    )
    
    conf_mat %>% autoplot(type = "heatmap")
    
}

### Test the function ----
classify_with_c5_trees(
    .model           = "decision_tree",
    .mode            = "classification",
    .engine          = "C5.0",
    .tree_depth      = NULL, # for decision_tree
    .trees           = NULL, # for boost_tree
    .min_n           = 1  # for both, NULL produces error, so > 1 is adviced
)
