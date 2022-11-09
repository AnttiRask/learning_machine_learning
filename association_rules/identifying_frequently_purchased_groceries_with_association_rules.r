# Identifying Frequently Purchased Groceries with Association Rules ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 8:
# Finding Patterns - Market Basket Analysis Using Association Rules
#
# The original code is made with {arules}. Now, this is an algorithm that isn't
# found in {tidymodels}. I still wanted to recreate it using {tidyverse} as much
# as possible. Helps with the readability if nothing else!
#
# You can find the original code and the dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter08

## 1. Loading libraries ----
library(conflicted)
library(tidyverse)
library(arules)
library(arulesViz)


## 2. Exploring and preparing the data ----

### Load the grocery data into a sparse matrix ----
groceries <- read.transactions(
    "association_rules/data/groceries.csv",
    sep = ","
)
summary(groceries)

### Look at the first five transactions ----
groceries %>%
    head(5) %>%
    inspect()

### Examine the frequency of items ----
groceries[, 1:3] %>%
    itemFrequency()

### Plot the frequency of items ----
groceries %>%
    itemFrequencyPlot(support = 0.1)

groceries %>%
    itemFrequencyPlot(topN = 20)

### A visualization of the sparse matrix for the first five transactions ----
groceries[1:5] %>%
    image()

### Visualization of a random sample of 100 transactions ----
sample(groceries, 100) %>%
    image()


## 3. Training a model on the data ----

### Default settings result in zero rules learned ----
groceries %>%
    apriori()

### Set better support and confidence levels to learn more rules ----
groceryrules <- apriori(
    groceries,
    parameter = list(
        support = 0.006,
        confidence = 0.25,
        minlen = 2
    )
)
groceryrules


## 4. Evaluating model performance ----

# Summary of grocery association rules ----
groceryrules %>% summary()

# Look at the first three rules ----
groceryrules[1:3] %>%
    inspect()

# Visualize the rules ----
groceryrules %>% plot(
    method = "graph",
    limit  = 5
)


## 5. Improving model performance ----

### Sorting grocery rules by lift ----
sort(groceryrules, by = "lift")[1:5] %>%
    inspect()

### Finding subsets of rules containing any berry items ----
berryrules <- groceryrules %>%
    subset(items %in% "berries")

berryrules %>%
    inspect()

### Converting the rule set to a tibble ----
groceryrules_tbl <- groceryrules %>%
    as("data.frame") %>%
    as_tibble()

groceryrules_tbl %>%
    glimpse()
