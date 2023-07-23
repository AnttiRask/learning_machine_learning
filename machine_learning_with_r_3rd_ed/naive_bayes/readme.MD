# Filtering Mobile Phone Spam with Naive Bayes

Inspired by Brett Lantz's Machine Learning with R,<br>
Chapter 4: Probabilistic Learning - Classification Using Naive Bayes and<br>
Chapter 10: Evaluating Model Performance.

The original code is made with a lot of base R, {e1071} and {gmodels}. I wanted to see how one could recreate it using mainly {textrecipes}, {tidymodels}, {tidytext} and {tidyverse}.

If you haven't already, buy the book: https://www.packtpub.com/product/machine-learning-with-r-third-edition/9781788295864

You can find the original code and the slightly modified dataset here:<br>
https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter04<br>
https://github.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/tree/master/Chapter10

Dataset Citation:
[1] Gómez Hidalgo, J.M., Cajigas Bringas, G., Puertas Sãnz, E., Carrero García, F. Content Based SMS Spam Filtering. Proceedings of the 2006 ACM Symposium on Document Engineering (ACM DOCENG'06), Amsterdam, The Netherlands, 10-13, 2006.

[2] Cormack, G. V., Gómez Hidalgo, J. M., and Puertas Sãnz, E. Feature engineering for mobile (SMS) spam filtering. Proceedings of the 30th Annual international ACM Conference on Research and Development in information Retrieval (ACM SIGIR'07), New York, NY, 871-872, 2007.

[3] Cormack, G. V., Gómez Hidalgo, J. M., and Puertas Sãnz, E. Spam filtering for short messages. Proceedings of the 16th ACM Conference on Information and Knowledge Management (ACM CIKM'07). Lisbon, Portugal, 313-320, 2007.

Also, there were parts that were inspired by:

A blog post by Julia Silge:
https://www.tidyverse.org/blog/2020/11/tidymodels-sparse-support/

Julia Silge & David Robinson's book Text Mining with R: A Tidy Approach: https://www.tidytextmining.com/

A blog post by Céline Van den Rul:
https://towardsdatascience.com/create-a-word-cloud-with-r-bde3e7422e8a