# Predicting Medical Expenses with Linear Regression ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 6:
# Forecasting Numeric Data - Regression Methods.
#
# The original code is made with base R and {psych}. I wanted to see how one
# could recreate it using mainly {tidymodels} and {tidyverse}.
#
# You can find the original code and the dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter06

## 1. Loading libraries ----
library(tidyverse)
library(corrr)
library(tidymodels)
library(GGally)


## 2. Exploring and preparing the data ----
insurance_tbl <- read_csv("regression_methods/data/insurance.csv") 
glimpse(insurance_tbl)

### Summarize the data ----
insurance_tbl %>%
  mutate(across(where(is.character), as_factor)) %>% 
  summary()

### Histogram of insurance charges ----
insurance_tbl %>% 
  ggplot(aes(expenses)) +
  geom_histogram(binwidth = 5000)

### Distribution between regions ----
insurance_tbl %>%
  count(region) %>% 
  mutate(pct = (n / sum(n) * 100))

### Exploring relationships among features: correlation matrix ----
insurance_tbl %>%
  select(c("age", "bmi", "children", "expenses")) %>% 
  as.matrix() %>% 
  correlate()

### Visualing relationships among features: scatterplot matrix ----
insurance_tbl %>%
  select(c("age", "bmi", "children", "expenses")) %>%
  ggpairs()


## 3. Training a model on the data ----

### Create a recipe ----
recipe_obj <- recipe(
  expenses ~ .,
  data = insurance_tbl
)
recipe_obj

insurance_baked_tbl <- recipe_obj %>%
  prep() %>%
  bake(new_data = NULL)
insurance_baked_tbl

### Model specification ----
model_spec <- linear_reg(
  mode            = "regression",
  engine          = "lm"
) %>%
  translate()
model_spec

### Fit the model ----
model_fit <- fit(
  model_spec,
  expenses ~ .,
  insurance_baked_tbl
)

### See the estimated beta coefficients ----
model_fit


## 4. Evaluating model performance ----
summary(model_fit$fit)


## 5. Improving model performance ----

### Add a higher-order "age" term ----
insurance_augmented_tbl <- insurance_baked_tbl %>% 
  mutate(age2 = age ^ 2,
         
         ### Add an indicator for BMI >= 30 ----
         bmi30 = 
           case_when(
             bmi >= 30 ~ 1,
             TRUE      ~ 0
           )
  )


### Create the final model ----
recipe_augmented_obj <- recipe(
  expenses ~ .,
  data = insurance_augmented_tbl
) %>% 
  # This step is needed for the next step, adding the interaction between bmi30 and smoker
  step_dummy(smoker) %>%
  # Dummy variables need to be specified with starts_with()
  step_interact(terms = ~ bmi30:starts_with("smoker"))
recipe_augmented_obj

insurance_baked_augmented_tbl <- recipe_augmented_obj %>%
  prep() %>%
  bake(new_data = NULL)
insurance_baked_augmented_tbl

### Model specification ----
model_spec_augmented <- linear_reg(
  mode            = "regression",
  engine          = "lm"
) %>%
  translate()
model_spec_augmented

### Fit the model ----
model_fit_augmented <- fit(
  model_spec_augmented,
  expenses ~ .,
  insurance_baked_augmented_tbl
)

### See the estimated beta coefficients ----
model_fit_augmented

### Evaluate model performance ----
summary(model_fit_augmented$fit)

### Make predictions with the regression model (you could skip this step) ----
insurance_pred_tbl <- predict(
  object   = model_fit_augmented,
  new_data = insurance_baked_augmented_tbl,
  type     = "numeric"
)
insurance_pred_tbl

### Add the predictions to the tibble ----
insurance_pred_tbl <- augment(model_fit_augmented, insurance_baked_augmented_tbl)
insurance_pred_tbl

### See the correlation between the actual and predicted expenses ----
insurance_pred_tbl %>%
  select(.pred, expenses) %>% 
  correlate()

# Or with the more simple vectorized version:
cor(insurance_pred_tbl$.pred, insurance_pred_tbl$expenses)

### Visualize the correlation ----
insurance_pred_tbl %>%
  ggplot(aes(.pred, expenses)) +
  geom_point() +
  geom_abline(
    intercept = 0,
    slope     = 1,
    color     = "red",
    size      = 0.5,
    linetype  = "dashed" 
  ) +
  labs(
    x = "Predicted Expenses",
    y = "Actual Expenses"
  )

### See the model metrics ----
insurance_model_metrics <- insurance_pred_tbl %>% 
  metrics(
    truth    = expenses,
    estimate = .pred
  )
insurance_model_metrics


### Make predictions with new data ----
predict(
  model_fit_augmented,
  tibble(
    age                = 30,
    age2               = 30^2,
    bmi                = 30,
    bmi30              = 1,
    bmi30_x_smoker_yes = 0,
    children           = 2,
    region             = "northeast",
    sex                = "male",
    smoker_yes         = 0
  )
)

predict(
  model_fit_augmented,
  tibble(
    age                = 30,
    age2               = 30^2,
    bmi                = 30,
    bmi30              = 1,
    bmi30_x_smoker_yes = 0,
    children           = 2,
    region             = "northeast",
    sex                = "female",
    smoker_yes         = 0
  )
)

predict(
  model_fit_augmented,
  tibble(
    age                = 30,
    age2               = 30^2,
    bmi                = 30,
    bmi30              = 1,
    bmi30_x_smoker_yes = 0,
    children           = 0,
    region             = "northeast",
    sex                = "female",
    smoker_yes         = 0
  )
)


## 6. Creating a function to help evaluate the model further ----

# The assumption here is that you have already gone through steps 1. to 5.
# You will see the predicted medical expenses, if you enter the following
# parameters: age, BMI, number of children, region, sex and whether you smoke
# or not.

predict_medical_expenses <- function(
  .age,
  .bmi,
  .children,
  .region = c("northeast", "northwest", "southeast", "southwest"),
  .sex    = c("female, male"),
  .smoker = c("no", "yes")
){
  
  .bmi30 <- if_else(
    condition = .bmi >= 30,
    true      = 1,
    false     = 0,
    missing   = NULL
  )
  
  .smoker_yes <- if_else(
    condition = .smoker == "yes",
    true      = 1,
    false     = 0,
    missing   = NULL
  )
  
  .bmi30_x_smoker_yes <- if_else(
    condition = .bmi30 == TRUE && .smoker_yes == TRUE,
    true      = 1,
    false     = 0,
    missing   = NULL
  )
  
  prediction <- predict(
    model_fit_augmented,
    tibble(
      age                = .age,
      age2               = .age ^ 2,
      bmi                = .bmi,
      bmi30              = .bmi30,
      bmi30_x_smoker_yes = .bmi30_x_smoker_yes,
      children           = .children,
      region             = .region,
      sex                = .sex,
      smoker_yes         = .smoker_yes
    )
  )
  
  str_glue("The predicted medical expenses according to the given parameters are: ${prediction %>% round(0)}")
  
}

### Test the function ----
predict_medical_expenses(
  .age      = 30,
  .bmi      = 30,
  .children = 0,
  .region   = "northeast",
  .sex      = "female",
  .smoker   = "no"
)