# Markov Random Fields

We studied undirected probabilistic graphical models and here we provide some result in image processing.

We sampled Markov Random Fields models (Ising model, Potts model) using the following algorithm:
* Gibbs sampling
* Metropolis sampling
* Iterated Conditional Modes (ICM)

Finally we applied these methods to solve a task of image denoising.

## Ising model
We aim to sample the following distribution on state space {0,1}^n where

<img src="img/gibbs_eq.png" width="60%">

<img src="img/ising_eq.png" width="30%">

We achieved sampling Ising model by using 3 different algorithms: Gibbs sampling, Metropolis sampling, ICM.

  Example with beta = 1.  :

<img src="img/ising_gibbs_1_.png" width="60%">

  Example with beta = 3.  :

<img src="img/ising_gibbs_3_.png" width="60%">

While MCMC algorithms Gibbs and Metropolis have theorical guarantee to converge to the designed distribution, ICM converge quickly to a local minimum. On the following graph we compare convergence of these algorithms by measuring how fast they minimize the global energy of the distribution.

<img src="img/all_curves.png" width="60%">

If ICM is faster it requires to start from a suitable initial solution and if not it may not converge at all. On the following plots we compare the ability of respectively ICM, Gibbs and Metropolis algorithms to converge to Ising model distribution starting from a all white image.

<img src="img/ising_icm_from_zeros_.png" width="60%">
<img src="img/ising_gibbs_from_zeros_.png" width="60%">
<img src="img/ising_metro_from_zeros_.png" width="60%">

## Potts model

We also sample the Potts model, a more generalized version of Ising model as the state space {0,1,2,...,q}^n is no more binary.

In the following we sampled:

<img src="img/potts_simple_eq.png" width="20%">

<img src="img/potts_no_mix_.png" width="60%">

In the following we sampled:

<img src="img/potts_hierarch_eq.png" width="35%">

<img src="img/potts_with_hierarch_mix_.png" width="60%">

## Image denoising 

In order to perform image denoising using Markov Random Fields we added some random gaussian noise to an original binary image. We then aim to recover the original image. We show that the use of an Ising prior achieves better result than a naive independancy prior.

<img src="img/duo_test.png" width="60%">

We performed image denoising in the bayesian framework using a naive prior assumption of independancy between pixels then using a prior following Ising distribution with 4 connexity. 

Naive prior:

<img src="img/naive_prior.png" width="30%">

Ising prior with 4 connexity:

<img src="img/ising_prior.png" width="40%">

The Likelhood (data attachment) is modelled by a gaussian. Hence we have: 

<img src="img/likelyhood.png" width="50%">
<img src="img/likely_norm.png" width="55%">

We use MAP in order to recover original image:

<img src="img/map.png" width="55%">

First we estimated parameters of gaussian by computing empirical mean and standard deviation on random patches of the image. We also computed confidence intervals to ensure of the accuracy of the estimators. 

<img src="img/stats_patches.png" width="40%">

We then applied ICM algorithm to solve MAP. Actually **with naive prior** it is equivalent to simple thresholding.

<img src="img/test_without_.png" width="90%">

We then applied ICM algorithm to solve MAP **with Ising prior** and we see that the final image obtained is closer to the orginal image than in the previous case.

<img src="img/test_with_ising_.png" width="90%">

This observation is confirmed by the following plot which highlights how the two models converge to original image.

<img src="img/recovery_curves.png" width="60%">
