# Model selection

In the previous chapters, we considered models (Ridge, LASSO) that have parameters. In data science, we often consider whole families of models that can depend on millions or even billions of parameters. In this chapter, we will address how to rigorously select the best model from a set of possible candidate. The method of choice for accomplishing that task is **cross-validation**. 

## Cross-validation

In [Lab 2](sec-lab2), we discussed the difference between the training and the test error of a model. Recall that during training, we attempt to select the model parameters that fit the data well. However, this does not immediately guarantee that the model will do well on new data. In order to estimate the error made on unseen data, we therefore split our data into a training set and a test set. We train the model on the training set, and then measure its error on the test set. Cross-validation pushes this idea even further in the training phase to better train the model to make good predictions on new data. This approach can be used to select the parameters of models such as Ridge and LASSO regression. 

To perform cross validation, we first split our training data into $K$ subsets of roughly equal size. Each subset is called a *fold*. Next, for each fold, we fit the model on the training data, *excluding* the current fold. We then measure the error of the fitted model in the current fold. This is similar to our train/test approach from before. However, we perform that process several time to get a better estimate of the error made by the model on unseen data. 


```{figure} images/cv.png
---
width: 500 px
---
In cross-validation, one fold is used for validation while the other folds are used for training.
```


