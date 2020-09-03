# Minibatch Wasserstein distance

Python3 implementation of the paper [Learning with minibatch Wasserstein: asymptotic and gradient properties
](https://arxiv.org/abs/1910.04091) (AISTATS 2020)

Optimal transport distances are powerful tools to compare probability distributions and have found many applications in machine learning. Yet their algorithmic complexity prevents their direct use on large scale datasets. To overcome this challenge, practitioners compute these distances on minibatches, i.e. they average the outcome of several smaller optimal transport problems. We propose in this paper an analysis of this practice, which effects are not well understood so far. We notably argue that it is equivalent to an implicit regularization of the original problem, with appealing properties such as unbiased estimators, gradients and a concentration bound around the expectation, but also with defects such as loss of distance property. Along with this theoretical analysis, we also conduct empirical experiments on gradient flows, GANs or color transfer that highlight the practical interest of this strategy.

We also wrote a [medium blog post](https://medium.com/p/learning-with-minibatch-wasserstein-d87dcf52efb5?source=email-d0d7857135bb--writer.postDistributed&sk=4c30efd3442780edf7ca140080557476), feel free to ask if any question.

If you use this toolbox in your research or minibatch Wasserstein and find them useful, please cite minibatch Wasserstein using the following bibtex reference:

```
@InProceedings{fatras2019learnwass,
author    = {Fatras, Kilian and Zine, Younes and Flamary, Rémi and Gribonval, Rémi and Courty, Nicolas},
title     = {Learning with minibatch Wasserstein: asymptotic and gradient properties},
booktitle = {AISTATS},
year      = {2020 (To appear)}
}
```

### Prerequisites

* Numpy (>= 1.11)
* Matplotlib (>= 1.5)
* time
* sklearn
* For Optimal transport [Python Optimal Transport](https://pot.readthedocs.io/en/stable/) POT (>=0.5.1)


### What is included ?

* Minibatch Wasserstein color transfer (large scale)
* Deviation of mb OT matrix and marginals
* Time experiment
* Slides
* Poster


### Authors

* [Kilian Fatras](https://kilianfatras.github.io/)
* [Younès Zine](https://www.linkedin.com/in/youn%C3%A8s-zine-7abb68149/?originalSubdomain=fr)
* [Rémi Flamary](http://remi.flamary.com/)
* [Rémi Gribonval](http://people.irisa.fr/Remi.Gribonval/)
* [Nicolas Courty](https://github.com/ncourty)


## References

[1] Flamary Rémi and Courty Nicolas [POT Python Optimal Transport library](https://github.com/rflamary/POT)
