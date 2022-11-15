# Finding Teen Market Segments Using k-means Clustering

Inspired by Brett Lantz's Machine Learning with R, Chapter 9: Finding Groups of Data - Clustering with k-means.

The original code is made with base R. While {tidymodels} doesn't have a k-means engine, I wanted to see how one could still recreate the code using {tidyverse} as much as possible and even using the {recipes} package from the {tidymodels} to do all the pre-processing needed and {broom} package from the same {tidymodels} family to help look at the model metrics in a more tidy way.

I was also inspired by this blog post from the tidymodels website to continue the exercise a bit further than the original code did:
https://www.tidymodels.org/learn/statistics/k-means/

If you haven't already, buy the book: https://www.packtpub.com/product/machine-learning-with-r-third-edition/9781788295864

You can find the original code and the dataset here:
https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter09

Dataset Citation:
Brett Lantz. "Finding Groups of Data - Clustering with k-means". (Machine Learning with R)