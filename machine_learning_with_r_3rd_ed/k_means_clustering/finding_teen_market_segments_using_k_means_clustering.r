# Finding Teen Market Segments Using k-means Clustering ----

# Inspired by Brett Lantz's Machine Learning with R, Chapter 9:
# Finding Groups of Data - Clustering with k-means
#
# The original code is made with base R. While {tidymodels} doesn't have a
# k-means engine, I wanted to see how one could still recreate the code using
# {tidyverse} as much as possible and even using the {recipes} package from the
# {tidymodels} to do all the pre-processing needed and {broom} package from the
# same {tidymodels} family to help look at the model metrics in a more tidy way.
#
# I was also inspired by this blog post from the tidymodels website to continue
# the exercise a bit further than the original code did:
# https://www.tidymodels.org/learn/statistics/k-means/
#
# You can find the original code and the dataset here:
# https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter09

## 1. Loading libraries ----
library(conflicted)
library(tidyverse)
library(janitor)
library(tidymodels)


## 2. Exploring and preparing the data ----
teens_tbl <- read_csv("machine_learning_with_r_3rd_ed/k_means_clustering/data/snsdata.csv")

teens_tbl %>%
    glimpse()

### Look at missing data for age variable ----
teens_tbl %>%
    select(age) %>%
    summary()

### Eliminate age outliers ----
teens_only_tbl <- teens_tbl %>%
    mutate(
        age = case_when(
            between(age, 13, 20) ~ age,
            TRUE      ~ NA_real_
        )
    )

teens_only_tbl %>%
    select(age) %>%
    summary()

### Look at missing data for female variable ----
teens_only_tbl %>%
    tabyl(gender)

### Reassign missing gender values to "unknown" and change into factors ----
teens_gendered_tbl <- teens_only_tbl %>%
    mutate(
        gender = case_when(
            gender %in% c("F", "M") ~ gender,
            TRUE                    ~ "gender_unknown"
        ) %>%
            as.factor()
    )

### Check our recoding work ----
teens_gendered_tbl %>%
    tabyl(gender) %>%
    arrange(desc(percent))

### Finding the mean age by cohort ----
# Doesn't work because of the NAs
teens_gendered_tbl %>%
    pull(age) %>%
    mean()

# Works thanks to the na.rm = TRUE
teens_gendered_tbl %>%
    pull(age) %>%
    mean(na.rm = TRUE)

### Age by cohort ----
teens_gendered_tbl %>%
    with_groups(
        gradyear,
        summarize,
        age = mean(age, na.rm = TRUE)
    )

### Create a vector with the average age for each gradyear, repeated by person ----
average_age_by_gradyear <- teens_gendered_tbl %>%
    with_groups(
        gradyear,
        mutate,
        ave_age = mean(age, na.rm = TRUE)
    ) %>%
    pull()

### Impute the missing age values with the average age by gradyear ----
teens_imputed_tbl <- teens_gendered_tbl %>%
    mutate(
        age = case_when(
            !is.na(age) ~ age,
            TRUE        ~ average_age_by_gradyear
        )
    )

### Check the summary results to ensure missing values are eliminated ----
teens_imputed_tbl %>%
    select(age) %>%
    summary()


## 3. Creating the recipe ----

### Apply normalization to entire data frame ----
recipe_obj <- recipe(teens_imputed_tbl) %>%
    step_normalize(
        all_numeric(),
        -c(gradyear, age, friends)
    ) %>%
    step_dummy(
        gender,
        keep_original_cols = TRUE,
        one_hot            = TRUE
    )
recipe_obj

recipe_baked_tbl <- recipe_obj %>%
    prep() %>%
    bake(new_data = NULL)
recipe_baked_tbl


## 4. Training a model on the data ----

### Create a z-score standardized data frame for easier interpretation ----
interests_tbl <- recipe_baked_tbl %>%
    select(5:40)

### Compare the data before and after the transformation ----

# Before
teens_imputed_tbl %>%
    select(basketball) %>%
    summary()

# After
interests_tbl %>%
    select(basketball) %>%
    summary()

### Create the clusters using k-means ----
RNGversion("3.5.2")
set.seed(2345)

teens_clusters <- interests_tbl %>%
    kmeans(5)
teens_clusters


## 5. Evaluating model performance ----

### Look at the single-row summary ----
teens_clusters %>%
    glance()

### Look at the size and the centers of the clusters ----
teens_clusters %>%
    tidy() %>%
    select(cluster, size, withinss, everything())


## 6. Improving model performance ----

### Apply the cluster IDs to the original data frame ----
teens_and_clusters <- augment(teens_clusters, recipe_baked_tbl)
teens_and_clusters

### Look at the first five records ----
teens_and_clusters %>%
    select(.cluster, gender, age, friends) %>%
    slice_head(n = 5)

### Mean age by cluster ----
teens_and_clusters %>%
    with_groups(
        .cluster,
        summarize,
        age = mean(age)
    )

### Proportion of females by cluster ----
teens_and_clusters %>%
    with_groups(
        .cluster,
        summarize,
        gender_F = mean(gender_F)
    )

### Mean number of friends by cluster ----
teens_and_clusters %>%
    with_groups(
        .cluster,
        summarize,
        friends = mean(friends)
    )


## 7. K-means clustering with tidy data principles ----

### Exploratory clustering ----
kclusts <-
    tibble(k = 1:9) %>%
    mutate(
        kclust = map(k, ~kmeans(interests_tbl, .x)),
        tidied = map(kclust, tidy),
        glanced = map(kclust, glance),
        augmented = map(kclust, augment, recipe_baked_tbl)
    )

### Create three separate datasets ----
clusters <- kclusts %>%
    unnest(cols = c(tidied))

assignments <- kclusts %>%
    unnest(cols = c(augmented))

clusterings <- kclusts %>%
    unnest(cols = c(glanced))

### Plot the original points ----
p1 <- assignments %>%
    ggplot(aes(sports, music)) +
    geom_point(aes(color = .cluster), alpha = 0.5) +
    facet_wrap(vars(k)) +
    theme_bw()
p1

p2 <- p1 +
    geom_point(data = clusters, size = 5, shape = "x")
p2


### Plot the total within sum of squares (tot.withinss) ----
clusterings %>%
    ggplot(aes(k, tot.withinss)) +
    geom_line() +
    geom_point() +
    theme_bw() +
    scale_x_continuous(breaks = seq(1, 9, by = 1)) +
    labs(
        x = "k",
        y = "Total Within Sum of Squares"
    )
