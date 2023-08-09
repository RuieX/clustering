# Assignment 3: Clustering

Perform classification of the [MNIST database](http://yann.lecun.com/exdb/mnist/) (or a sufficiently small subset of it) using:

* mixture of Gaussian's with diagonal covariance (Gaussian Naive Bayes with latent class label);
* mean shift;
* normalized cut.

The unsupervised classification must be performed at **varying levels of dimensionality reduction through PCA**  (say going from 2 to 200) 
In order to assess the effect of the **dimensionality** in **accuracy and learning time**.

Provide the _code_ and 
the _extracted clusters_ as the number of **clusters k varies from 5 to 15**, for the mixture of Gaussian's and normalized-cut, 
while for mean shift **vary the kernel width**. 
_For each_ value of k (or kernel width) _provide the value of the Rand index_:

R=2(a+b)/(n(n-1))
$$
R=2\frac{a+b}{n(n-1)}
$$

where
* $n$ is the number of images in the dataset.
* $a$ is the number of pairs of images that represent the same digit and that are clustered together.
* $b$ is the number of pairs of images that represent different digits and that are placed in different clusters.

Explain the differences between the three models.

**Tip**: 
the means of the Gaussian models can be visualized as a greyscale images after PCA reconstruction to inspect the learned model.