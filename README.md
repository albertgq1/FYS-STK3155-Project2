Project 2 in FYS-STK3155/4155 (Applied Data Analysis and Machine Learning, University of Oslo, autumn 2026).
Author: Alberto García Asensio.

We implement a dense feed-forward neural network (FFNN) from scratch with NumPy, trained with
backpropagation and plain gradient descent, RMSprop or Adam, and apply it to two problems:

1. **Regression** of the noisy one-dimensional Runge function, f(x) = 1/(1+25x^2), the same data as in
   Project 1. The network is compared with OLS, Ridge and Lasso polynomial regression.
2. **Classification** of the MNIST handwritten digits (10 classes), compared with logistic regression
   and with scikit-learn's `MLPClassifier`.

We study the learning rate and optimiser, the number of hidden layers and nodes, the activation function
(sigmoid, ReLU, Leaky ReLU), overfitting and vanishing gradients, and L1 and L2 penalties. All
hyperparameters are chosen with 5-fold cross-validation (regression) or a validation set (MNIST);
the test sets are used only once at the end.

Main results: the network reaches a cross-validated MSE of about 0.012 on the Runge function (noise
level 0.01), comparable to degree-12 OLS and Ridge but stable at the edges of the interval; on MNIST a
network with one hidden layer of 200 ReLU nodes reaches 97.7 % test accuracy (logistic regression
92.7 %, `MLPClassifier` 97.8 %).


# How to run

Requirements: Python 3 with `numpy`, `matplotlib` and `scikit-learn`.

```
pip install numpy matplotlib scikit-learn
```

Open `Code/Project2.ipynb` and run all cells from the top (the notebook must be in the same folder as
`mnist.npz`). All results use the random seed 2026, so the numbers in the report are reproduced. The
figures are saved as PDF files in the same folder as the notebook.
