# Boosting with tidymodels ----

# Inspired by Brett Lantz's Machine Learning with R, 
# Chapter 5: Divide and Conquer - Classification Using Decision Trees and Rules and
# Chapter 10: Evaluating Model Performance
#
# The original code is made with {C50}, {gmodels}, {OneR} and {RWeka}. I
# wanted to see how one could recreate it using mainly {tidymodels} and
# {tidyverse}.
#
# You can find the original code and the slightly modified dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter05
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter10

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


## 4. Building a boosted ensemble ----

### Specify the model class ----
boost_spec <- boost_tree() %>%
    # Set the mode
    set_mode("classification") %>%
    # Set the engine
    set_engine("xgboost")
boost_spec


## 5. Evaluate the ensemble ----
set.seed(99)

### Create CV folds ----
folds <- vfold_cv(data = credit_train, v = 10)
folds

### Fit and evaluate models for all folds ----
cv_results <- fit_resamples(
    boost_spec,
    default ~ .,
    resamples = folds,
    metrics   = metric_set(roc_auc)
)

### Collect cross-validated metrics ----
collect_metrics(cv_results)


## 6. Tuning preparation ----

### Create the specification with placeholders ----
boost_spec <- boost_tree(
    trees = 500,
    learn_rate = tune(),
    tree_depth = tune(),
    sample_size = tune()) %>%
    set_mode("classification") %>%
    set_engine("xgboost")

### Create the tuning grid ----
tunegrid_boost <- grid_regular(
    extract_parameter_set_dials(boost_spec), 
    levels = 3
)
tunegrid_boost


## 7. The actual tuning ----

### Create CV folds of training data ----
folds <- vfold_cv(
    credit_train,
    v = 10
)

### Tune along the grid ----
tune_results <- tune_grid(
    boost_spec,
    default ~ .,
    resamples = folds,
    grid      = tunegrid_boost,
    metrics   = metric_set(roc_auc)
)

### Plot the results ----
autoplot(tune_results)


## 8. Finalizing the model ----

### Select the final hyperparameters ----
best_params <- select_best(tune_results)
best_params

### Finalize the specification ----
final_spec <- finalize_model(boost_spec, best_params)
final_spec

### Train the final model on the full training data ----
final_model <- final_spec %>%
    fit(
        formula = default ~ .,
        data = credit_train
    )
final_model

JATKA TÄSTÄ!
https://app.datacamp.com/learn/courses/machine-learning-with-tree-based-models-in-r

# ### Train the forest ----
# model <- spec %>%
#     fit(
#         default ~ .,
#         data = credit_train
#     )
# model
# 
# ### Plot the variable importance
# vip::vip(model)
# 
# 
# ## 5. Predicting ----
# 
# ### Make the predictions (you could skip this step) ----
# credit_test_pred <- predict(
#     object   = model,
#     new_data = credit_test,
#     type     = "class"
# )
# credit_test_pred
# 
# ### Add the predictions to the test tibble ----
# credit_test_with_pred_tbl <- augment(model, credit_test)
# credit_test_with_pred_tbl
# 
# 
# ## 6. Evaluating model performance ----
# 
# ### Create a confusion matrix ----
# conf_mat <- conf_mat(
#     data     = credit_test_with_pred_tbl,
#     truth    = default,
#     estimate = .pred_class
# )
# conf_mat
# 
# ### Visualize the confusion matrix ----
# conf_mat %>% autoplot(type = "heatmap")
# conf_mat %>% autoplot(type = "mosaic")
# 
# ### Visualize the ROC curve ----
# credit_test_with_pred_tbl %>%
#     roc_curve(
#         truth    = default,
#         estimate = .pred_no
#     ) %>%
#     autoplot()
# 
# ### Calculate the ROC AUC (area under the curve) ----
# credit_roc_auc <- credit_test_with_pred_tbl %>%
#     roc_auc(
#         truth    = default,
#         estimate = .pred_no
#     )
# credit_roc_auc
# 
# ### Put together other model metrics ----
# # Such as accuracy, Matthews correlation coefficient (mcc) and others...
# classification_metrics <- conf_mat(
#     credit_test_with_pred_tbl,
#     truth    = default,
#     estimate = .pred_class
# ) %>%
#     summary()
# classification_metrics
