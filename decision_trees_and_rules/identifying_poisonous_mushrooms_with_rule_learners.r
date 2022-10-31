# Identifying Poisonous Mushrooms with Rule Learners ----

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
library(tidyverse)
library(tidymodels)
library(rules)

## 2. Exploring and preparing the data ----
mushrooms_tbl <- read_csv("decision_trees_and_rules/data/mushrooms.csv")

### Examine the data ----
mushrooms_tbl %>% map(unique)

### Drop the veil_type feature ----
mushroom_selected_tbl <- mushrooms_tbl %>%
    select(-veil_type)

### Examine the class distribution ----
mushroom_selected_tbl %>%
    count(type) %>%
    mutate(pct = (n / sum(n) * 100))


## 3. Creating the recipe ----
recipe_obj <- recipe(
    type ~ .,
    data = mushroom_selected_tbl
)
recipe_obj

mushroom_baked_tbl <- recipe_obj %>%
    prep() %>%
    bake(new_data = NULL)


## 4. Training a model on the data ----

### Model specification ----
model_spec <- C5_rules(
    mode            = "classification",
    engine          = "C5.0",
    trees           = NULL,
    min_n           = NULL
) %>%
    translate()
model_spec

### Fit the model ----
model_fit <- fit(
    model_spec,
    type ~ .,
    mushroom_baked_tbl
)
model_fit

model_fit %>%
    extract_fit_engine() %>%
    summary()

### Make the predictions (you could skip this step) ----
mushroom_pred_tbl <- predict(
    object   = model_fit,
    new_data = mushroom_baked_tbl,
    type     = "class"
)
mushroom_pred_tbl

### Add the predictions to the test tibble ----
mushroom_pred_tbl <- augment(model_fit, mushroom_baked_tbl)
mushroom_pred_tbl


## 5. Evaluating model performance ----

### Create a confusion matrix ----
conf_mat <- conf_mat(
    data     = mushroom_pred_tbl,
    truth    = type,
    estimate = .pred_class
)
conf_mat

### Visualize the confusion matrix ----
conf_mat %>% autoplot(type = "heatmap")
conf_mat %>% autoplot(type = "mosaic")

### Visualize the ROC curve ----
mushroom_pred_tbl %>%
    roc_curve(
        truth    = type,
        estimate = .pred_edible
    ) %>%
    autoplot()

### Calculate the ROC AUC (area under the curve) ----
mushroom_roc_auc <- mushroom_pred_tbl %>%
    roc_auc(
        truth    = type,
        estimate = .pred_edible
    )
mushroom_roc_auc

### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...
classification_metrics <- conf_mat(
    mushroom_pred_tbl,
    truth    = type,
    estimate = .pred_class
) %>%
    summary()
classification_metrics


## 6. Creating a function to help evaluate the model further ----

# The assumption here is that you have already gone through steps 1. to 4.
# What we're potentially tuning here are the arguments .trees and .min_n

classify_with_c5_rules <- function(
    .trees           = NULL,
    .min_n           = NULL
) {
    
    # Create the recipe
    recipe_obj <- recipe(
        type ~ .,
        data = mushroom_selected_tbl
    )
    
    mushroom_baked_tbl <- recipe_obj %>%
        prep() %>%
        bake(new_data = NULL)
    
    # Model specification
    model_spec <- C5_rules(
        mode            = "classification",
        engine          = "C5.0",
        trees           = .trees,
        min_n           = .min_n
    ) %>%
        translate()
    
    # Fit the model
    model_fit <- fit(
        model_spec,
        type ~ .,
        mushroom_baked_tbl
    )
    
    model_fit %>%
        extract_fit_engine() %>%
        summary()
    
    # Add the predictions to the test tibble
    mushroom_pred_tbl <- augment(model_fit, mushroom_baked_tbl)
    mushroom_pred_tbl
    
    # Create a confusion matrix
    conf_mat <- conf_mat(
        data     = mushroom_pred_tbl,
        truth    = type,
        estimate = .pred_class
    )
    
    conf_mat %>% autoplot(type = "heatmap")
    
}

### Test the function ----
classify_with_c5_rules(
    .trees = 3,
    .min_n = 1
)
