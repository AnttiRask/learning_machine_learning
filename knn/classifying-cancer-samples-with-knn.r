# Classifying Cancer Samples with K-Nearest Neighbors ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 3:
# Lazy Learning - Classification Using Nearest Neighbors.
#
# The original code is made with a lot of base R, {class} and {gmodels}. I
# wanted to see how one could recreate it using mainly {tidymodels}
# and {tidyverse}.
#
# You can find the original code here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter03

## 1. Loading libraries ----
library(tidymodels)
library(tidyverse)


## 2. Exploring and preparing the data ----

### Create a vector for the column names ----
.col_names <- c(
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "points_mean",
    "symmetry_mean",
    "dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "points_se",
    "symmetry_se",
    "dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "points_worst",
    "symmetry_worst",
    "dimension_worst"
)

### Import the CSV file (Breast Cancer Wisconsin (Diagnostic)) ----
wbcd_tbl <- read_csv("knn/data/wdbc-data.csv", col_names = .col_names)

### Take a look at the tibble ----
glimpse(wbcd_tbl)

### Drop the unnecessary id column ----
wbcd_selected_tbl <- wbcd_tbl %>% select(-id)
wbcd_selected_tbl

### Transform diagnosis to a factor ----
wbcd_factored_tbl <- wbcd_selected_tbl %>%
    mutate(
        diagnosis = factor(
            diagnosis,
            levels = c("B", "M"),
            labels = c("Benign", "Malignant")
        )
    )
wbcd_factored_tbl

### Count the number of the two diagnosis (incl. percentage) ----
wbcd_factored_tbl %>%
    count(diagnosis) %>%
    mutate(pct = (n / sum(n) * 100))

### Summarize three numeric features ----
wbcd_factored_tbl %>%
    select(radius_mean, area_mean, smoothness_mean) %>%
    summary()


## 3. Creating the recipe and splitting the data ----

### Normalize the wbcd data ----
recipe_obj <- recipe(
    diagnosis ~ .,
    data = wbcd_factored_tbl
) %>%
    step_range(
        all_numeric_predictors(),
        min = 0,
        max = 1
    )
recipe_obj

wbcd_normalized_tbl <- recipe_obj %>%
    prep() %>%
    bake(new_data = NULL)

### Confirm that normalization worked ----
wbcd_normalized_tbl %>%
    select(area_mean) %>%
    summary()

### Create training and test data (randomly) ----
wbcd_split <- initial_split(
    wbcd_normalized_tbl,
    prop = 469 / 569
)
wbcd_train <- training(wbcd_split)
wbcd_test  <- testing(wbcd_split)


## 4. Training a model on the data ----

# kknn is the engine (needs to be installed if not already):
# install.packages("kknn")

# It is used as the engine for {parsnip}'s nearest_neighbor() function.
# And since we are classifying, that is the mode we choose.

### Create model specification ----
model_spec <- nearest_neighbor(
    engine      = "kknn",
    mode        = "classification",
    neighbors   = 21
) %>%
    translate()
model_spec

### Fit the model ----
model_fit <- fit(
    model_spec,
    diagnosis ~ .,
    wbcd_train
)
model_fit

### Make the predictions (you could skip this step) ----
wbcd_test_pred <- predict(
    model_fit,
    new_data = wbcd_test,
    type = "class"
)
wbcd_test_pred

### Add the predictions to the test tibble ----
wbcd_test_with_pred_tbl <- augment(model_fit, wbcd_test)
wbcd_test_with_pred_tbl


## 5. Evaluating model performance ----

### Create a confusion matrix ----
conf_mat <- conf_mat(
    data     = wbcd_test_with_pred_tbl,
    truth    = diagnosis,
    estimate = .pred_class
)
conf_mat

### Visualize the confusion matrix ----
conf_mat %>% autoplot(type = "heatmap")
conf_mat %>% autoplot(type = "mosaic")

### Visualize the ROC curve ----
wbcd_test_with_pred_tbl %>%
    roc_curve(
        truth    = diagnosis,
        estimate = .pred_Benign
    ) %>%
    autoplot()

### Calculate the ROC AUC (area under the curve) ----
wbcd_roc_auc <- wbcd_test_with_pred_tbl %>%
    roc_auc(
        truth    = diagnosis,
        estimate = .pred_Benign
    )
wbcd_roc_auc


### Put together other model metrics ----
# Such as accuracy, Matthews correlation coefficient (mcc) and others...
classification_metrics <- conf_mat(
    wbcd_test_with_pred_tbl,
    truth    = diagnosis,
    estimate = .pred_class
) %>%
    summary()


## 6. Creating a function to help evaluate the model further ----

# The assumption here is that you have already gone through steps 1. to 2.
# The idea is to be able to choose different values for k and different
# methods for standardization (range (0 to 1) and normalization)

classify_with_knn <- function(
    k = 21,
    standardization_method = c("range", "normalization")
) {
    
    # Create a recipe according to the chosen standardization method
    if (standardization_method == "range") {
        
        recipe_obj <- recipe(
            formula = diagnosis ~ .,
            data    = wbcd_factored_tbl
        ) %>%
            step_range(
                all_numeric_predictors(),
                min = 0,
                max = 1)
        
    } else if (standardization_method == "normalization") {
        
        recipe_obj <- recipe(
            formula = diagnosis ~ .,
            data    = wbcd_factored_tbl
        ) %>%
            step_normalize(all_numeric_predictors())
        
    } else {
        
        stop('Choose a starndardization method that is either "range" or "normalization"!')
        
    }
    
    wbcd_normalized_tbl <- recipe_obj %>%
        prep() %>%
        bake(new_data = wbcd_factored_tbl)
    
    # Create training and test data
    wbcd_split <- initial_split(
        wbcd_normalized_tbl,
        prop = 469 / 569
    )
    wbcd_train <- training(wbcd_split)
    wbcd_test  <- testing(wbcd_split)
    
    # Create model specification
    model_spec <- nearest_neighbor(
        engine      = "kknn",
        mode        = "classification",
        neighbors   = k
    ) %>%
        translate()
    
    # Fit the model
    model_fit <- fit(
        model_spec,
        diagnosis ~ .,
        wbcd_train
    )
    
    # Add the predictions to the test tibble
    wbcd_test_with_pred_tbl <- augment(model_fit, wbcd_test)
    
    # Create a confusion matrix
    conf_mat <- conf_mat(
        data     = wbcd_test_with_pred_tbl,
        truth    = diagnosis,
        estimate = .pred_class
    )
    
    # Print the confusion matrix
    conf_mat %>% autoplot(type = "heatmap")
}

### Test the function ----
classify_with_knn(
    standardization_method = "range",
    k = 5
)
